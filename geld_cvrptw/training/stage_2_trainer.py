"""
class that runs the stage 2 training procedure with supervised learning labels.
"""

from logging import getLogger

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.distributions.uniform import Uniform

from geld_cvrptw.data.benchmark_loaders import records_to_benchmark_instances
from geld_cvrptw.data.generator import generate_instances, _generate_valid_batch
from geld_cvrptw.inference.decoders.beam_search import BeamSearchDecoder
from geld_cvrptw.inference.pipeline import InferencePipeline, build_pipeline
from geld_cvrptw.inference.types import SolveResult
from geld_cvrptw.utils.experiment_tracker import ExperimentTracker, should_log_batch
from geld_cvrptw.utils.metrics import AverageMeter, LogData, TimeEstimator
from geld_cvrptw.utils.logging import (
    get_result_folder,
    util_print_log_array,
)
from geld_cvrptw.utils.device import setup_device


from geld_cvrptw.model.GELD_CVRPTW import GeldCvrptwModel
from geld_cvrptw.env.CVRPTW import CVRPTWEnv
from geld_cvrptw.data.loaders import TOUR_PAD_VALUE, load_cvrptw_data_with_labels
from geld_cvrptw.model.helpers import teacher_action_prob, apply_feasibility_mask


class Stage2Trainer:
    """
    Stage 2 trainer with curriculum learning / Self Improvement Learning 
    Model is trained on instances of increasing difficulty (size)
    """

    def __init__(
        self,
        env_params: dict,
        model_params: dict,
        optimizer_params: dict,
        trainer_params: dict,
        tracker: ExperimentTracker | None = None,
    ):

        # Setup logging
        self.logger = getLogger(name="trainer")
        self.result_folder = get_result_folder()
        self.result_log = LogData()
        self.tracker = tracker

        # Where to run training
        self.device = setup_device(trainer_params["use_cuda"], trainer_params["cuda_device_num"])
        """Initializes model etc with hyperparams from config. """
        # Reproducability
        torch.manual_seed(42)

        # Save Params
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # Initialize model, env, optim + scheduler and set hyperparams
        self.model = GeldCvrptwModel(**self.model_params).to(self.device)
        self.env = CVRPTWEnv(**self.env_params)
        self.env.set_device(self.device)

        self.scheduler = MultiStepLR(self.optimizer, **self.optimizer_params["scheduler"])

        # Load the checkpoint from stage 1 training (SL) to continue
        checkpoint_path = f"{trainer_params['model_load_path']}/checkpoint-{trainer_params['model_load_epoch']}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer = Adam(self.model.parameters(), **self.optimizer_params["optimizer"])

        # Training restart (if stage 2 training already started)
        self.start_epoch = 1
        if self.trainer_params.get("model_load", {}).get("enable", False):
            self.continue_training_from_checkpoint()


        # Load BS decoder with standard params (16 Beams)
        self.solver:InferencePipeline = InferencePipeline(
            decoder = BeamSearchDecoder(),
            postprocessors = None
        )


        # Training time estimator
        self.time_estimator = TimeEstimator()

    def run_training(self):
        """Run stage 1 supervised training"""

        # Load Train Data into memory.
        self.env.set_device(self.device)

        # load the dataset with the small instanes  once 
        self.small_instances = self._load_small_samples()
        

        # Training with problem sizes from min to max (curriculum learning)
        problem_size_min = self.trainer_params["problem_size_min"]
        problem_size_max = self.trainer_params["problem_size_max"]
        self.logger.info(f"Starting curriculum leanring from size {problem_size_min} to {problem_size_max}")


        for epoch in range(self.start_epoch, self.trainer_params["epochs"] + 1):
            # Linear increase towards max size
            current_problem_size = problem_size_min + epoch * (problem_size_max - problem_size_min) // self.trainer_params["epochs"]

            train_reference_length, train_predicted_length, train_loss, learning_rate = self._train_one_epoch(epoch, current_problem_size)
            self.log_metrics(
                epoch,
                train_reference_length,
                train_predicted_length,
                train_loss,
                learning_rate,
            )
            self.save_model_checkpoints_and_progress(epoch)


    def _load_small_samples(self):
        """
        Loads and returns the stage 1 training dataset with size 100 problems
        """

        dataset = load_cvrptw_data_with_labels()

        coords = dataset["coords"].requires_grad_(False)
        demand = dataset["demand"].requires_grad_(False)
        tw_start = dataset["tw_start"].requires_grad_(False)
        tw_end = dataset["tw_end"].requires_grad_(False)
        service_time = dataset["service_time"].requires_grad_(False)
        tours = dataset["label_tours"].requires_grad_(False)
        costs = dataset["costs"].requires_grad_(False)

        return coords, demand, tw_start, tw_end, service_time, tours, costs



    def _generate_large_samples_with_pseudolabels(self, epoch:int, current_problem_size:int):
        """
        Generates a dataset of CVRPTW instances and returns them in the format that the inference pipeline needs
        """
                # Generate Random instances of current problem size
        alpha = Uniform(0.5, 3.0).sample().item()
        depot_xy, node_xy, node_demand, capacity, service_time, tw_start, tw_end = _generate_valid_batch(
            num_samples=self.trainer_params["train_episodes"],
            problem_size=current_problem_size,
            alpha=alpha,
            seed=epoch,
            batch_size=self.trainer_params["train_batch_size"],
        )

        # Normalize + Prepend depot at index 0
        coords = torch.cat((depot_xy, node_xy), dim=1)
        normalized_node_demand = node_demand / capacity[:, None]
        demand = torch.cat(
            (torch.zeros_like(capacity[:, None]), normalized_node_demand),
            dim=1,
        )
        tw_start = torch.cat((torch.zeros_like(capacity[:, None]), tw_start), dim=1)
        tw_end = torch.cat((torch.full_like(capacity[:, None], 3.0), tw_end), dim=1)
        service_time = torch.cat((torch.zeros_like(capacity[:, None]), service_time), dim=1)

        self.env.load_problem_tensors(coords, demand, tw_start, tw_end, service_time)
        solve_result:SolveResult = self.solver.run(self.model, self.env)
        self.env.reset()

        return coords, demand, tw_start, tw_end, service_time, solve_result.tour, solve_result.length_normalized

    def make_batches(self, instances,  batch_size:int, num_batches:int|None = None, shuffle:bool=True):
        """
        Splits instances into batches. instances expected to be:
        (coords, demand, tw_start, tw_end, service_time, tours, costs)
        """
        coords, demand, tw_start, tw_end, service_time, tours, costs = instances
        num_samples = coords.size(0)

        if shuffle:
            indices = torch.randperm(num_samples, device=coords.device)
        else:
            indices = torch.arange(num_samples, device=coords.device)

        batches = []
        for start in range(0, num_samples, batch_size):
            if num_batches is not None and len(batches) >= num_batches:
                return batches
            end = min(start + batch_size, num_samples)
            batch_idx = indices[start:end]
            batches.append(
                (
                    coords[batch_idx],
                    demand[batch_idx],
                    tw_start[batch_idx],
                    tw_end[batch_idx],
                    service_time[batch_idx],
                    tours[batch_idx],
                    costs[batch_idx],
                )
            )
        return batches


    def _train_one_epoch(self, epoch: int, current_problem_size:int):
        """
        1. Generates data of the current size (+ size 100 problems mixed in to not degrade preformance on small ones)
        2. Generate pseudo labels using self.solver (beam search decoder)
        3. With the pseude labels do the same SL as before 
        """

        large_instances = self._generate_large_samples_with_pseudolabels(epoch, current_problem_size)
        
        large_batches = self.make_batches(large_instances, batch_size=self.trainer_params["train_batch_size"], shuffle=True)
        num_batches = len(large_batches)
        small_batches = self.make_batches(self.small_instances, batch_size=self.trainer_params["train_batch_size"], num_batches=num_batches,  shuffle=True)
        
        if len(large_batches) != len(small_batches):
            raise ValueError(
                f"Expected matching batch counts, got large={len(large_batches)} and small={len(small_batches)}"
            )

        # Metrics Tracking over batches in epoch
        reference_length_meter = AverageMeter()
        predicted_length_meter = AverageMeter()
        loss_meter = AverageMeter()

        num_total_samples = sum(batch[0].size(0) for batch in large_batches) + sum(
            batch[0].size(0) for batch in small_batches
        )
        num_processed_samples = 0
        batch_log_interval = self.trainer_params["logging"].get("batch_log_interval", 50)

        # Alternate one large batch and one small batch.
        for large_batch, small_batch in zip(large_batches, small_batches):
            for current_batch in (large_batch, small_batch):
                batch_size = current_batch[0].size(0)
                reference_length, predicted_length, avg_loss = self._train_one_batch(current_batch)

                # Logging & Tracking
                reference_length_meter.update(reference_length, batch_size)
                predicted_length_meter.update(predicted_length, batch_size)
                loss_meter.update(avg_loss, batch_size)
                num_processed_samples += batch_size
                if should_log_batch(num_processed_samples, num_total_samples, batch_log_interval):
                    self.logger.info(
                        f"Epoch {epoch:3d}: Train {num_processed_samples:3d}/{num_total_samples:3d}"
                        f"({100.0 * num_processed_samples / num_total_samples:1.1f}%)  "
                        f"Reference length: {reference_length_meter.avg:.4f}, "
                        f"Predicted length: {predicted_length_meter.avg:.4f}, "
                        f"Loss: {loss_meter.avg:.4f}"
                    )

        learning_rate = self.optimizer.param_groups[0]["lr"]
        self.scheduler.step()

        # Get averages over all batches
        avg_reference_length = reference_length_meter.avg
        avg_predicted_length = predicted_length_meter.avg
        avg_loss = loss_meter.avg

        return avg_reference_length, avg_predicted_length, avg_loss, learning_rate

    def _train_one_batch(self, batch):
        """
        Train one batch supervised learning: cross entropy with teacher/ground truth.
        Each sample is stepped through its full label tour (variable length, depot returns).
        At each step of the teacher tour we ask the model where it would go next
        and compute cross entropy for the outputted probability distribution.
        """
        self.model.train()

        # Load one batch of samples
        coords, demand, tw_start, tw_end, service_time, tours, costs = batch
        self.env.load_batch_tensors(coords, demand, tw_start, tw_end, service_time, tours, costs, train=True)
        batch_size = coords.size(0)

        # step 1 is fixed, for tracking the models preds
        # We just track them for each step that we let it predict
        # we DONT use it to compute the joint probability of that tour under the model, only per step predictions.
        step_log_probs = torch.ones(size=(batch_size, 0), device=self.device)

        static_state, dynamic_state = self.env.reset()
        self.model.embed_static_state_once(static_state)

        label_tour = static_state.label_tour
        tour_lengths = (label_tour != TOUR_PAD_VALUE).sum(dim=1)

        current_step = 1
        while (current_step < tour_lengths).any():
            active = current_step < tour_lengths

            raw_probs = self.model(dynamic_state, mask_feasibility=False)
            probs = apply_feasibility_mask(raw_probs, dynamic_state.ninf_mask)
            
            teacher_node = label_tour[:, current_step]
            teacher_node = torch.where(active, teacher_node, label_tour[:, 0])
            predicted_node = probs.argmax(dim=1)
            predicted_node = torch.where(active, predicted_node, label_tour[:, 0])
            step_prob = teacher_action_prob(raw_probs, teacher_node).unsqueeze(1)
            step_prob = torch.where(active.unsqueeze(1), step_prob, torch.ones_like(step_prob))

            # Negative log-likelihood of the label action under the model distribution.
            loss_mean = -step_prob[active].type(torch.float64).log().mean()
            self.model.zero_grad()
            loss_mean.backward()
            self.optimizer.step()

            dynamic_state = self.env.step(teacher_node, predicted_node, dynamic_state)
            step_log_probs = torch.cat((step_log_probs, step_prob), dim=1)
            current_step += 1

        reference_length = self.env.compute_tour_length(self.env.batch_coords, static_state.label_tour).mean().item()
        # model_tour is label[1:]; prepend depot so length includes the first leg and matches label tour_lengths.
        depot_column = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
        predicted_tour = torch.cat((depot_column, dynamic_state.model_tour), dim=1) # include depot to first node
        predicted_length = (
            self.env.compute_tour_length(self.env.batch_coords, predicted_tour, tour_lengths=tour_lengths)
            .mean()
            .item()
        )
        loss_mean = -step_log_probs.log().mean()
        return reference_length, predicted_length, loss_mean.item()

    def num_train_samples(self) -> int:
        """Full loaded dataset size, optionally capped by train_episodes for debugging."""
        cap = self.trainer_params.get("train_episodes")
        dataset_size = self.env.num_samples()
        if cap is None:
            return dataset_size
        return min(dataset_size, cap)

    def log_metrics(
        self,
        epoch: int,
        train_reference_length: float,
        train_predicted_length: float,
        train_loss: float,
        learning_rate: float,
    ):
        """Does all the logging, tracking to wandb etc for one epoch"""
        self.logger.info("=================================================================")

        # Log
        self.result_log.append("train_reference_length", epoch, train_reference_length)
        self.result_log.append("train_predicted_length", epoch, train_predicted_length)
        self.result_log.append("train_loss", epoch, train_loss)
        self.result_log.append("learning_rate", epoch, learning_rate)

        elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params["epochs"])
        self.logger.info(
            f"Epoch {epoch:3d}/{self.trainer_params['epochs']:3d}: "
            f"ref={train_reference_length:.4f}, pred={train_predicted_length:.4f}, "
            f"loss={train_loss:.4f}, lr={learning_rate:.2e}, "
            f"Time Est.: Elapsed[{elapsed_time_str}], Remain[{remain_time_str}]"
        )
        if self.tracker is not None:
            self.tracker.log_epoch(
                {
                    "epoch": epoch,
                    "train_reference_length": train_reference_length,
                    "train_predicted_length": train_predicted_length,
                    "train_loss": train_loss,
                    "learning_rate": learning_rate,
                },
                step=epoch,
            )
            self.tracker.save_training_progress(
                self.result_log,
                metadata={
                    "run_type": "train_sl",
                    "trainer_params": self.trainer_params,
                },
                save_plots=epoch > 1,
            )

    def save_model_checkpoints_and_progress(self, epoch: int):
        """
        Saves checkpoints in intervals as well as training progress at the checkpoints and when training is done.
        """
        model_save_interval = self.trainer_params["logging"]["model_save_interval"]
        # Training done?
        all_done = epoch == self.trainer_params["epochs"]

        if all_done or (epoch % model_save_interval) == 0:
            self.logger.info("Saving trained_model")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "result_log": self.result_log.get_raw_data(),
                },
                f"{self.result_folder}/checkpoint-{epoch}.pt",
            )

        if all_done:
            self.logger.info(" *** Training Done *** ")
            util_print_log_array(self.logger, self.result_log)
            if self.tracker is not None:
                self.tracker.save_training_progress(
                    self.result_log,
                    metadata={
                        "run_type": "train_sl",
                        "trainer_params": self.trainer_params,
                    },
                )
                self.tracker.finish()

    def continue_training_from_checkpoint(self):
        """
        continues training from a checkpoint according to config.
        Loads params and model, optim etc. state
        """
        model_load = self.trainer_params["model_load"]
        checkpoint_path = f"{model_load['path']}/checkpoint-{model_load['epoch']}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.start_epoch = 1 + model_load["epoch"]
        self.result_log.set_raw_data(checkpoint["result_log"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.last_epoch = model_load["epoch"] - 1
        self.logger.info("Saved Model Loaded !!")
