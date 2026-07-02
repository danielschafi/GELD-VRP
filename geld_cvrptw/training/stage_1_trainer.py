"""
class that runs the stage 1 training procedure with supervised learning labels.
"""

from geld_cvrptw.data.loaders import iter_offset_batches
from geld_cvrptw.training.base_trainer import BaseTrainer
from geld_cvrptw.utils.experiment_tracker import ExperimentTracker, should_log_batch
from geld_cvrptw.utils.metrics import AverageMeter


class Stage1Trainer(BaseTrainer):
    """Stage 1 trainer. Trains on small instances (100 customers)."""

    def __init__(
        self,
        env_params: dict,
        model_params: dict,
        optimizer_params: dict,
        trainer_params: dict,
        tracker: ExperimentTracker | None = None,
    ):
        super().__init__(
            env_params,
            model_params,
            optimizer_params,
            trainer_params,
            run_type="train_stage_1",
            tracker=tracker,
        )

    def run_training(self):
        """Run stage 1 supervised training"""

        # Load Train Data into memory.
        self.env.load_all_data()
        self.env.set_device(self.device)

        for epoch in range(self.start_epoch, self.trainer_params["epochs"] + 1):
            train_reference_length, train_predicted_length, train_loss, learning_rate = self._train_one_epoch(epoch)
            self.log_metrics(
                epoch,
                train_reference_length,
                train_predicted_length,
                train_loss,
                learning_rate,
            )
            self.save_model_checkpoints_and_progress(epoch)

    def _train_one_epoch(self, epoch: int):
        """One train pass over all training instances."""
        self.env.shuffle_full_data()

        # Metrics Tracking over batches in epoch
        reference_length_meter = AverageMeter()
        predicted_length_meter = AverageMeter()
        loss_meter = AverageMeter()

        num_total_samples = self.num_train_samples()
        num_processed_samples = 0
        batch_log_interval = self.trainer_params["logging"].get("batch_log_interval", 50)
        batch_size = self.trainer_params["batch_size"]

        for batch_offset, current_batch_size in iter_offset_batches(num_total_samples, batch_size):
            # Training
            reference_length, predicted_length, avg_loss = self._train_one_batch(batch_offset, current_batch_size)

            # Logging & tracking
            reference_length_meter.update(reference_length, current_batch_size)
            predicted_length_meter.update(predicted_length, current_batch_size)
            loss_meter.update(avg_loss, current_batch_size)
            num_processed_samples += current_batch_size
            if should_log_batch(num_processed_samples, num_total_samples, batch_log_interval):
                self.logger.info(
                    f"Epoch {epoch:3d}: Train {num_processed_samples:3d}/{num_total_samples:3d}"
                    f"({100.0 * num_processed_samples / num_total_samples:1.1f}%)  "
                    f"Reference length: {reference_length_meter.avg:.4f}, "
                    f"Predicted length: {predicted_length_meter.avg:.4f}, "
                    f"Loss: {loss_meter.avg:.4f}"
                )

        return self._finalize_epoch(reference_length_meter, predicted_length_meter, loss_meter)

    def _train_one_batch(self, batch_offset: int, batch_size: int):
        """Load one dataset batch and run supervised training."""
        self.env.load_batch_from_dataset(batch_offset, batch_size, train=True)
        return self._train_supervised_batch(batch_size)
