"""Stage 2 curriculum trainer with greedy/beam/PRC label generation."""

from logging import getLogger

import torch
from torch.optim import Adam as Optimizer

from geld.env.synthetic import SyntheticEnvironment
from geld.model.geld_model import GeldModel
from geld.search.prc import apply_prc_iteration
from geld.search.solver import InferenceSolver
from geld.utils.device import setup_device
from geld.utils.experiment_tracker import ExperimentTracker, should_log_batch
from geld.utils.logging import get_result_folder, util_print_log_array
from geld.utils.metrics import AverageMeter, LogData, TimeEstimator


class CurriculumTrainer:
    """Stage-2 SIL trainer with curriculum scaling and BS/PRC pseudo-labels."""

    def __init__(
        self,
        env_params,
        model_params,
        optimizer_params,
        trainer_params,
        tracker: ExperimentTracker | None = None,
    ):
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        self.logger = getLogger(name="trainer")
        self.result_folder = get_result_folder()
        self.result_log = LogData()
        self.tracker = tracker
        self.device = setup_device(trainer_params["use_cuda"], trainer_params["cuda_device_num"])

        torch.manual_seed(2024)
        self.model = GeldModel(**self.model_params).to(self.device)
        self.env = SyntheticEnvironment(**self.env_params)
        self.env.set_device(self.device)

        checkpoint_path = f"{trainer_params['model_load_path']}/checkpoint-{trainer_params['model_load_epoch']}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params["optimizer"])

        self.start_epoch = 1
        model_load = trainer_params["model_load"]
        if model_load["enable"]:
            resume_path = f"{model_load['path']}/checkpoint-{model_load['epoch']}.pt"
            checkpoint = torch.load(resume_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.start_epoch = 1 + model_load["epoch"]
            self.result_log.set_raw_data(checkpoint["result_log"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.logger.info("Saved Model Loaded !!")

        self.solver = InferenceSolver(
            self.model,
            self.env,
            self.device,
            use_beam=True,
            use_prc=False,
            beam_size=trainer_params["beam_size"],
        )
        self.time_estimator = TimeEstimator()

    def run(self):
        """Run curriculum SIL from TSP-k_m toward n_max."""
        self.time_estimator.reset(self.start_epoch)
        curriculum_episodes = self.trainer_params.get("curriculum_data_episodes", 1_000_000)
        self.env.load_raw_data(curriculum_episodes, load_eval_data=False)
        self.env.set_device(self.device)
        problem_size_init = self.trainer_params["problem_size_init"]
        problem_size_max = self.trainer_params["problem_size_max"]

        for epoch in range(self.start_epoch, self.trainer_params["epochs"] + 1):
            self.logger.info("=================================================================")
            problem_size = (
                problem_size_init + epoch * (problem_size_max - problem_size_init) // self.trainer_params["epochs"]
            )
            (
                train_reference_length,
                train_predicted_length,
                train_loss,
                greedy_mean_length,
                best_mean_length,
            ) = self._train_one_epoch(epoch, problem_size)
            self.result_log.append("problem_size", epoch, problem_size)
            self.result_log.append("train_reference_length", epoch, train_reference_length)
            self.result_log.append("train_predicted_length", epoch, train_predicted_length)
            self.result_log.append("train_loss", epoch, train_loss)
            self.result_log.append("greedy_mean_length", epoch, greedy_mean_length)
            self.result_log.append("best_mean_length", epoch, best_mean_length)

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params["epochs"])
            self.logger.info(
                f"Epoch {epoch:3d}/{self.trainer_params['epochs']:3d}: "
                f"n={problem_size}, ref={train_reference_length:.4f}, pred={train_predicted_length:.4f}, "
                f"loss={train_loss:.4f}, greedy={greedy_mean_length:.4f}, best={best_mean_length:.4f}, "
                f"Time Est.: Elapsed[{elapsed_time_str}], Remain[{remain_time_str}]"
            )
            if self.tracker is not None:
                self.tracker.log_epoch(
                    {
                        "epoch": epoch,
                        "problem_size": problem_size,
                        "train_reference_length": train_reference_length,
                        "train_predicted_length": train_predicted_length,
                        "train_loss": train_loss,
                        "greedy_mean_length": greedy_mean_length,
                        "best_mean_length": best_mean_length,
                    },
                    step=epoch,
                )
                self.tracker.save_training_progress(
                    self.result_log,
                    logging_config=self.trainer_params.get("logging"),
                    metadata={
                        "run_type": "train_stage2",
                        "trainer_params": self.trainer_params,
                    },
                    save_plots=epoch > 1,
                )

            if (
                epoch == self.trainer_params["epochs"]
                or (epoch % self.trainer_params["logging"]["model_save_interval"]) == 0
            ):
                self.logger.info("Saving trained_model")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "result_log": self.result_log.get_raw_data(),
                    },
                    f"{self.result_folder}/checkpoint-{epoch}.pt",
                )

            if epoch == self.trainer_params["epochs"]:
                self.logger.info(" *** Training Done *** ")
                util_print_log_array(self.logger, self.result_log)
                if self.tracker is not None:
                    self.tracker.save_training_progress(
                        self.result_log,
                        logging_config=self.trainer_params.get("logging"),
                        metadata={
                            "run_type": "train_stage2",
                            "trainer_params": self.trainer_params,
                        },
                    )
                    self.tracker.finish()

    def _train_one_epoch(self, epoch, problem_size=100):
        """Generate pseudo-labels via BS/PRC, then iterate SL until convergence."""
        reference_length_meter = AverageMeter()
        predicted_length_meter = AverageMeter()
        loss_meter = AverageMeter()
        train_num_episode = self.trainer_params["train_episodes"]
        episode = 0
        batch_log_interval = self.trainer_params["logging"].get("batch_log_interval", 50)

        self.env.generate_random_instances(self.trainer_params["train_episodes"], problem_size)
        greedy_lengths, greedy_tours = self._validation_greedy()
        beam_lengths, beam_tours = self._validation_beam(problem_size)
        use_greedy = greedy_lengths < beam_lengths
        beam_tours[use_greedy] = greedy_tours[use_greedy]
        mean_prc_length, improved_tours = self._run_prc_training(beam_tours)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        best_mean_length = mean_prc_length
        self.env.raw_data_tours = improved_tours
        iterations = 0
        best_limit = self.trainer_params["best_limit"]

        while (
            iterations < self.trainer_params["max_limit"]
            and (greedy_lengths.mean() - best_mean_length) / best_mean_length > 0.001
            and best_limit > 0
        ):
            if best_mean_length > mean_prc_length:
                best_mean_length = mean_prc_length
                self.env.raw_data_tours = improved_tours
                best_limit = self.trainer_params["best_limit"]
            best_limit -= 1
            self.logger.info(
                f"Greedy mean length: {greedy_lengths.mean():.3f}, Best mean length: {best_mean_length:.3f}"
            )
            self.logger.info(
                f"iteration: {iterations:2d}, gap: {(greedy_lengths.mean() - best_mean_length) / best_mean_length:.3f}"
            )

            for _ in range(self.trainer_params["per_batch"]):
                while episode < train_num_episode:
                    remaining = train_num_episode - episode
                    batch_size = min(self.trainer_params["train_batch_size"], remaining)
                    reference_length, predicted_length, avg_loss = self._train_one_batch(
                        episode, batch_size, mix_curriculum_sizes=True
                    )
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    reference_length_meter.update(reference_length, batch_size)
                    predicted_length_meter.update(predicted_length, batch_size)
                    loss_meter.update(avg_loss, batch_size)
                    episode += batch_size
                    if should_log_batch(episode, train_num_episode, batch_log_interval):
                        self.logger.info(
                            f"Epoch {epoch:3d}: Train {episode:3d}/{train_num_episode:3d}"
                            f"({100.0 * episode / train_num_episode:1.1f}%)  "
                            f"Reference length: {reference_length_meter.avg:.4f}, "
                            f"Predicted length: {predicted_length_meter.avg:.4f}, "
                            f"Loss: {loss_meter.avg:.4f}"
                        )
                self.env.shuffle_data()
                episode = 0

            greedy_lengths, greedy_tours = self._validation_greedy()
            beam_lengths, beam_tours = self._validation_beam(problem_size)
            use_greedy = greedy_lengths < beam_lengths
            beam_tours[use_greedy] = greedy_tours[use_greedy]
            mean_prc_length, improved_tours = self._run_prc_training(beam_tours)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            iterations += 1

        return (
            reference_length_meter.avg,
            predicted_length_meter.avg,
            loss_meter.avg,
            float(greedy_lengths.mean().item()),
            float(best_mean_length),
        )

    @torch.no_grad()
    def _run_prc_training(self, tours, num_iterations=None):
        """Apply PRC iterations to improve tour pseudo-labels."""
        if num_iterations is None:
            num_iterations = self.trainer_params.get("prc_training_iterations", 1000)
        self.model.eval()
        val_num_episode = self.trainer_params["train_episodes"]
        problem_size = self.env.raw_data_nodes.size(1)
        sample_max = problem_size // 4

        current_tours = tours
        for step in range(num_iterations):
            num_segments = torch.randint(low=2, high=sample_max + 1, size=[1])[0]
            max_subpath = problem_size // num_segments
            if step % 2 != 0:
                current_tours = torch.flip(current_tours, dims=[1])
            current_tours = current_tours.roll(dims=1, shifts=int(torch.randint(low=0, high=problem_size, size=[1])[0]))
            batch_tours = None
            tour_lengths = None
            episode = 0
            while episode < val_num_episode:
                remaining = val_num_episode - episode
                batch_size = min(self.trainer_params["val_beam_batch_size"], remaining)
                rotation_id = torch.randint(low=0, high=8, size=[1])[0].item()
                self.env.load_problems_val(episode, batch_size, rotation_id)
                origin_problem = self.env.problems
                batch_tours_slice = current_tours[episode : episode + batch_size]
                torch.arange(0, problem_size, step=problem_size // num_segments, dtype=torch.long)[
                    :num_segments
                ]
                repaired = apply_prc_iteration(
                    self.model,
                    self.env,
                    origin_problem,
                    batch_tours_slice,
                    problem_size,
                    num_segments,
                    max_subpath,
                )
                batch_tours = torch.cat((batch_tours, repaired), dim=0) if batch_tours is not None else repaired
                current_length = self.env.compute_tour_length(origin_problem, repaired)
                tour_lengths = (
                    torch.cat((tour_lengths, current_length), dim=0) if tour_lengths is not None else current_length
                )
                episode += batch_size
            current_tours = batch_tours
        return tour_lengths.mean(), current_tours

    @torch.no_grad()
    def _validation_beam(self, problem_size):
        """Beam search all training instances to obtain improved pseudo-labels."""
        self.model.eval()
        tour_lengths = None
        tours = None
        episode = 0
        val_num_episode = self.trainer_params["train_episodes"]
        while episode < val_num_episode:
            remaining = val_num_episode - episode
            batch_size = min(self.trainer_params["val_beam_batch_size"], remaining)
            lengths, batch_tours = self.solver.run_beam_on_coordinates(episode, batch_size, problem_size)
            tour_lengths = torch.cat((tour_lengths, lengths), dim=0) if tour_lengths is not None else lengths
            tours = torch.cat((tours, batch_tours), dim=0) if tours is not None else batch_tours
            episode += batch_size
        return tour_lengths, tours

    @torch.no_grad()
    def _validation_greedy(self):
        """Greedy decode all training instances for pseudo-label comparison."""
        self.model.eval()
        tour_lengths = None
        tours = None
        episode = 0
        val_num_episode = self.trainer_params["train_episodes"]
        while episode < val_num_episode:
            remaining = val_num_episode - episode
            batch_size = min(self.trainer_params["val_batch_size"], remaining)
            lengths, batch_tours = self.solver.run_greedy_on_coordinates(episode, batch_size)
            tour_lengths = torch.cat((tour_lengths, lengths), dim=0) if tour_lengths is not None else lengths
            tours = torch.cat((tours, batch_tours), dim=0) if tours is not None else batch_tours
            episode += batch_size
        return tour_lengths, tours

    def _train_one_batch(self, batch_offset, batch_size, mix_curriculum_sizes=False):
        """SL batch with optional TSP-k_m curriculum mixing."""
        self.model.train()
        self.env.load_problems(
            batch_offset,
            batch_size,
            mix_curriculum_sizes=mix_curriculum_sizes,
            train=True,
        )
        problem_size = self.env.problems.size(1)
        if mix_curriculum_sizes:
            batch_size = self.env.batch_size

        step_log_probs = torch.ones(size=(batch_size, 0), device=self.device)
        result = self.env.reset()
        self.model.prepare_instance(result.coordinates)
        current_step = 0

        while not result.done:
            if current_step == 0:
                teacher_node = self.env.label_tour[:, -1]
                predicted_node = self.env.label_tour[:, -1]
                step_prob = torch.ones(batch_size, 1, device=self.device)
            elif current_step == 1:
                teacher_node = self.env.label_tour[:, 0]
                predicted_node = self.env.label_tour[:, 0]
                step_prob = torch.ones(batch_size, 1, device=self.device)
            else:
                output = self.model(self.env.constructed_tour, self.env.label_tour, current_step)
                teacher_node = output.teacher_action
                predicted_node = output.predicted_action
                step_prob = output.step_prob
                if mix_curriculum_sizes and problem_size - current_step > 98:
                    step_prob = step_prob[batch_size // 2 :]
                    step_prob = step_prob.repeat(2, 1)
                filtered_prob = step_prob[step_prob > 1e-3]
                loss_mean = -filtered_prob.type(torch.float64).log().mean()
                self.model.zero_grad()
                loss_mean.backward()
                self.optimizer.step()

            current_step += 1
            result = self.env.step(teacher_node, predicted_node)
            step_log_probs = torch.cat((step_log_probs, step_prob), dim=1)

        reference_length = self.env.compute_tour_length(self.env.problems, self.env.constructed_tour).mean().item()
        predicted_length = self.env.compute_tour_length(self.env.problems, self.env.model_tour).mean().item()
        loss_mean = -step_log_probs.log().mean()
        return reference_length, predicted_length, loss_mean.item()
