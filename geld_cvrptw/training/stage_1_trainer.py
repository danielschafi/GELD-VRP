"""
class that runs the stage 1 training procedure with supervised learning labels.
"""

from logging import getLogger

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from geld_cvrptw.utils.experiment_tracker import ExperimentTracker, should_log_batch
from geld_cvrptw.utils.metrics import AverageMeter, LogData, TimeEstimator
from geld_cvrptw.utils.logging import (
    get_result_folder,
    util_print_log_array,
    util_save_log_image_with_label,
)
from geld_cvrptw.utils.device import setup_device


from geld_cvrptw.model.GELD_CVRPTW import GeldCvrptwModel
from geld_cvrptw.env.CVRPTW import CVRPTWEnv
from geld_cvrptw.data.loaders import TOUR_PAD_VALUE
from geld_cvrptw.model.helpers import teacher_action_prob


class Stage1Trainer:
    """Stage 1 trainer. Trains on small instances (100 customers)"""

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

        self.optimizer = Adam(self.model.parameters(), **self.optimizer_params["optimizer"])
        self.scheduler = MultiStepLR(self.optimizer, **self.optimizer_params["scheduler"])

        # Training restart / continue handling
        self.start_epoch = 1
        if self.trainer_params.get("model_load", {}).get("enable", False):
            self.continue_training_from_checkpoint()

        # Training time estimator
        self.time_estimator = TimeEstimator()

    def run_training(self):
        """Run stage 1 supervised training"""

        # Load Train Data into memory.
        self.env.load_all_data()
        self.env.set_device(self.device)

        for epoch in range(self.start_epoch, self.trainer_params["epochs"] + 1):
            train_reference_length, train_predicted_length, train_loss = self._train_one_epoch(epoch)
            self.log_metrics(epoch, train_reference_length, train_predicted_length, train_loss)
            self.log_training_curve(epoch)
            self.save_model_checkpoints_and_progress(epoch)

    def _train_one_epoch(self, epoch: int):
        """One train pass over all training instances"""
        self.env.shuffle_full_data()

        # Metrics Tracking over batches in epoch
        reference_length_meter = AverageMeter()
        predicted_length_meter = AverageMeter()
        loss_meter = AverageMeter()

        num_total_samples = self.num_train_samples()
        num_processed_samples = 0
        batch_log_interval = self.trainer_params["logging"].get("batch_log_interval", 50)

        # Go over all training samples once
        while num_processed_samples < num_total_samples:
            remaining_samples = num_total_samples - num_processed_samples
            batch_size = min(self.trainer_params["train_batch_size"], remaining_samples)

            # Process Batch
            reference_length, predicted_length, avg_loss = self._train_one_batch(num_processed_samples, batch_size)

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

        self.scheduler.step()

        # Get averages over all batches
        avg_reference_length = reference_length_meter.avg
        avg_predicted_length = predicted_length_meter.avg
        avg_loss = loss_meter.avg

        return avg_reference_length, avg_predicted_length, avg_loss

    def _train_one_batch(self, batch_offset: int, batch_size: int):
        """
        Train one batch supervised learning: cross entropy with teacher/ground truth.
        Each sample is stepped through its full label tour (variable length, depot returns).
        At each step of the teacher tour we ask the model where it would go next
        and compute cross entropy for the outputted probability distribution.
        """
        self.model.train()

        # Load one batch of samples
        self.env.load_one_batch_of_problems(batch_offset, batch_size, train=True)

        # step 1 is fixed, for tracking the models preds
        # We just track them for each step that we let it predict
        # we DONT use it to compute the joint probability of that tour under the model, only per step predictions.
        step_log_probs = torch.ones(size=(batch_size, 0), device=self.device)

        static_state, dynamic_state = self.env.reset()
        self.model.embed_static_state_once(static_state)

        label_tour = static_state.label_tour
        tour_lengths = (label_tour != TOUR_PAD_VALUE).sum(dim=1)

        current_step = 0
        while (current_step < tour_lengths).any():
            active = current_step < tour_lengths

            if current_step == 0:
                # append last in cyclic (this is a node before return to depot)
                batch_idx = torch.arange(batch_size, device=self.device)
                last_idx = tour_lengths - 1
                teacher_node = label_tour[batch_idx, last_idx]
                predicted_node = teacher_node
                step_prob = torch.ones(batch_size, 1, device=self.device)
            elif current_step == 1:
                # append depot
                teacher_node = label_tour[:, 0]
                predicted_node = label_tour[:, 0]
                step_prob = torch.ones(batch_size, 1, device=self.device)
            else:
                probs = self.model(dynamic_state)
                teacher_node = label_tour[:, current_step - 1]
                predicted_node = probs.argmax(dim=1)
                step_prob = teacher_action_prob(probs, teacher_node).unsqueeze(1)
                teacher_node = torch.where(active, teacher_node, label_tour[:, 0])
                predicted_node = torch.where(active, predicted_node, label_tour[:, 0])

                # Negative log-likelihood of the label action under the model distribution.
                loss_mean = -step_prob[active].type(torch.float64).log().mean()
                self.model.zero_grad()
                loss_mean.backward()
                self.optimizer.step()

            dynamic_state = self.env.step(teacher_node, predicted_node)
            step_log_probs = torch.cat((step_log_probs, step_prob), dim=1)
            current_step += 1

        reference_length = self.env.compute_tour_length(self.env.batch_coords, static_state.label_tour).mean().item()
        predicted_length = (
            self.env.compute_tour_length(self.env.batch_coords, dynamic_state.model_tour, tour_lengths=tour_lengths)
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
    ):
        """Does all the logging, tracking to wandb etc for one epoch"""
        self.logger.info("=================================================================")

        # Log
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

    def log_training_curve(self, epoch: int):
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

    def save_model_checkpoints_and_progress(self, epoch: int):
        """
        Saves checkpoints in intervals as well as training progress at the checkpoints and when training is done.
        """
        model_save_interval = self.trainer_params["logging"]["model_save_interval"]
        img_save_interval = self.trainer_params["logging"]["img_save_interval"]
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
