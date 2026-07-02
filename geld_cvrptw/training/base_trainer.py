"""Shared training infrastructure for stage 1 and stage 2 trainers."""

from logging import getLogger

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from geld_cvrptw.data.loaders import TOUR_PAD_VALUE
from geld_cvrptw.env.CVRPTW import CVRPTWEnv
from geld_cvrptw.model.GELD_CVRPTW import GeldCvrptwModel
from geld_cvrptw.model.helpers import apply_feasibility_mask, teacher_action_prob
from geld_cvrptw.utils.experiment_tracker import ExperimentTracker
from geld_cvrptw.utils.logging import get_result_folder, util_print_log_array
from geld_cvrptw.utils.metrics import AverageMeter, LogData, TimeEstimator


class BaseTrainer:
    """Base class with shared init, supervised batch step, logging, and checkpoints."""

    def __init__(
        self,
        env_params: dict,
        model_params: dict,
        optimizer_params: dict,
        trainer_params: dict,
        run_type: str,
        tracker: ExperimentTracker | None = None,
    ):
        self.logger = getLogger(name="trainer")
        self.result_folder = get_result_folder()
        self.result_log = LogData()
        self.tracker = tracker
        self.run_type = run_type

        self.device = torch.device("cpu")
        self._setup_device(trainer_params)
        torch.manual_seed(42)

        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        env_init_params = dict(env_params)
        env_init_params["device"] = self.device
        self.env = CVRPTWEnv(**env_init_params)
        self.env.set_device(self.device)
        self.model = GeldCvrptwModel(**model_params).to(self.device)
        self.optimizer = Adam(self.model.parameters(), **optimizer_params["optimizer"])
        self.scheduler = MultiStepLR(self.optimizer, **optimizer_params["scheduler"])

        self.start_epoch = 1
        if self.trainer_params.get("resume_checkpoint", {}).get("enable", False):
            self.continue_training_from_checkpoint()

        self.time_estimator = TimeEstimator()

    def _setup_device(self, trainer_params: dict) -> None:
        from geld_cvrptw.utils.device import setup_device

        self.device = setup_device(trainer_params["use_cuda"], trainer_params["cuda_device_num"])

    def _train_supervised_batch(self, batch_size: int) -> tuple[float, float, float]:
        """
        Train one batch supervised learning: cross entropy with teacher/ground truth.
        Assumes the env batch is already loaded.
        """
        self.model.train()

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

            loss_mean = -step_prob[active].type(torch.float64).log().mean()
            self.model.zero_grad()
            loss_mean.backward()
            self.optimizer.step()

            dynamic_state = self.env.step(teacher_node, predicted_node, dynamic_state)
            step_log_probs = torch.cat((step_log_probs, step_prob), dim=1)
            current_step += 1

        reference_length = self.env.compute_tour_length(self.env.batch_coords, static_state.label_tour).mean().item()
        depot_column = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
        predicted_tour = torch.cat((depot_column, dynamic_state.model_tour), dim=1)
        predicted_length = (
            self.env.compute_tour_length(self.env.batch_coords, predicted_tour, tour_lengths=tour_lengths)
            .mean()
            .item()
        )
        loss_mean = -step_log_probs.log().mean()
        return reference_length, predicted_length, loss_mean.item()

    def _finalize_epoch(
        self,
        reference_length_meter: AverageMeter,
        predicted_length_meter: AverageMeter,
        loss_meter: AverageMeter,
    ) -> tuple[float, float, float, float]:
        learning_rate = self.optimizer.param_groups[0]["lr"]
        self.scheduler.step()
        return (
            reference_length_meter.avg,
            predicted_length_meter.avg,
            loss_meter.avg,
            learning_rate,
        )

    def num_train_samples(self) -> int:
        """Full loaded dataset size, optionally capped by instances_per_epoch for debugging."""
        cap = self.trainer_params.get("instances_per_epoch")
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
        """Log and track metrics for one epoch."""
        self.logger.info("=================================================================")

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
                    "run_type": self.run_type,
                    "trainer_params": self.trainer_params,
                },
                save_plots=epoch > 1,
            )

    def save_model_checkpoints_and_progress(self, epoch: int):
        """Save checkpoints at intervals and training progress when done."""
        model_save_interval = self.trainer_params["logging"]["model_save_interval"]
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
                        "run_type": self.run_type,
                        "trainer_params": self.trainer_params,
                    },
                )
                self.tracker.finish()

    def continue_training_from_checkpoint(self):
        """Continue training from a checkpoint according to config."""
        resume = self.trainer_params["resume_checkpoint"]
        checkpoint_path = f"{resume['path']}/checkpoint-{resume['epoch']}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.start_epoch = 1 + resume["epoch"]
        self.result_log.set_raw_data(checkpoint["result_log"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.last_epoch = resume["epoch"] - 1
        self.logger.info("Saved Model Loaded !!")
