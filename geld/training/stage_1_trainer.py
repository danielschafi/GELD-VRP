"""Stage 1 supervised learning trainer."""

from logging import getLogger

import torch
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from geld.env.synthetic import SyntheticEnvironment
from geld.model.geld_model import GeldModel
from geld.utils.device import setup_device
from geld.utils.experiment_tracker import ExperimentTracker, should_log_batch
from geld.utils.logging import (
    get_result_folder,
    util_print_log_array,
    util_save_log_image_with_label,
)
from geld.utils.metrics import AverageMeter, LogData, TimeEstimator


class TrainingStage1Trainer:
    """Stage-1 SL trainer on small-scale TSP-k_m instances."""

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
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params["optimizer"])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params["scheduler"])

        self.start_epoch = 1
        model_load = trainer_params["model_load"]
        if model_load["enable"]:
            checkpoint_path = f"{model_load['path']}/checkpoint-{model_load['epoch']}.pt"
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.start_epoch = 1 + model_load["epoch"]
            self.result_log.set_raw_data(checkpoint["result_log"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.last_epoch = model_load["epoch"] - 1
            self.logger.info("Saved Model Loaded !!")

        self.time_estimator = TimeEstimator()

    def run(self):
        """Run the full SL training loop with checkpointing."""
        self.time_estimator.reset(self.start_epoch)
        self.env.load_raw_data(self.trainer_params["train_episodes"])
        self.env.set_device(self.device)

        for epoch in range(self.start_epoch, self.trainer_params["epochs"] + 1):
            self.logger.info("=================================================================")
            self.env.shuffle_data()
            train_reference_length, train_predicted_length, train_loss = self._train_one_epoch(epoch)
            self.scheduler.step()

            self.result_log.append("train_reference_length", epoch, train_reference_length)
            self.result_log.append("train_predicted_length", epoch, train_predicted_length)
            self.result_log.append("train_loss", epoch, train_loss)

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params["epochs"])
            self.logger.info(
                f"Epoch {epoch:3d}/{self.trainer_params['epochs']:3d}: "
                f"ref={train_reference_length:.4f}, pred={train_predicted_length:.4f}, "
                f"loss={train_loss:.4f}, "
                f"Time Est.: Elapsed[{elapsed_time_str}], Remain[{remain_time_str}]"
            )
            if self.tracker is not None:
                self.tracker.log_epoch(
                    {
                        "epoch": epoch,
                        "train_reference_length": train_reference_length,
                        "train_predicted_length": train_predicted_length,
                        "train_loss": train_loss,
                    },
                    step=epoch,
                )
                self.tracker.save_training_progress(
                    self.result_log,
                    logging_config=self.trainer_params.get("logging"),
                    metadata={
                        "run_type": "train_sl",
                        "trainer_params": self.trainer_params,
                    },
                    save_plots=epoch > 1,
                )

            all_done = epoch == self.trainer_params["epochs"]
            model_save_interval = self.trainer_params["logging"]["model_save_interval"]
            img_save_interval = self.trainer_params["logging"]["img_save_interval"]

            if epoch > 1:
                image_prefix = f"{self.result_folder}/latest"
                util_save_log_image_with_label(
                    image_prefix,
                    self.trainer_params["logging"]["log_image_params_1"],
                    self.result_log,
                    labels=["train_reference_length"],
                )
                util_save_log_image_with_label(
                    image_prefix,
                    self.trainer_params["logging"]["log_image_params_2"],
                    self.result_log,
                    labels=["train_loss"],
                )

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

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = f"{self.result_folder}/img/checkpoint-{epoch}"
                util_save_log_image_with_label(
                    image_prefix,
                    self.trainer_params["logging"]["log_image_params_1"],
                    self.result_log,
                    labels=["train_reference_length"],
                )
                util_save_log_image_with_label(
                    image_prefix,
                    self.trainer_params["logging"]["log_image_params_2"],
                    self.result_log,
                    labels=["train_loss"],
                )

            if all_done:
                self.logger.info(" *** Training Done *** ")
                util_print_log_array(self.logger, self.result_log)
                if self.tracker is not None:
                    self.tracker.save_training_progress(
                        self.result_log,
                        logging_config=self.trainer_params.get("logging"),
                        metadata={
                            "run_type": "train_sl",
                            "trainer_params": self.trainer_params,
                        },
                    )
                    self.tracker.finish()

    def _train_one_epoch(self, epoch):
        """Train one epoch over all LEHD episodes."""
        reference_length_meter = AverageMeter()
        predicted_length_meter = AverageMeter()
        loss_meter = AverageMeter()
        train_num_episode = self.trainer_params["train_episodes"]
        episode = 0
        batch_log_interval = self.trainer_params["logging"].get("batch_log_interval", 50)

        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params["train_batch_size"], remaining)
            reference_length, predicted_length, avg_loss = self._train_one_batch(episode, batch_size)
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

        return reference_length_meter.avg, predicted_length_meter.avg, loss_meter.avg

    def _train_one_batch(self, batch_offset, batch_size):
        """One batch: teacher-forced cross-entropy over MDP steps."""
        self.model.train()
        self.env.load_problems(batch_offset, batch_size, train=True)
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
                loss_mean = -step_prob.type(torch.float64).log().mean()
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
