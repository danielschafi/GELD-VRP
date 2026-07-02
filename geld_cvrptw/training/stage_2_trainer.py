"""
class that runs the stage 2 training procedure with supervised learning labels.
"""

from logging import getLogger

import torch
from torch.distributions.uniform import Uniform

from geld_cvrptw.data.generator import generate_instance_tensors
from geld_cvrptw.data.loaders import TOUR_PAD_VALUE, iter_tensor_batches, load_cvrptw_data_with_labels
from geld_cvrptw.inference.pipeline import build_pipeline
from geld_cvrptw.inference.types import SolveResult
from geld_cvrptw.training.base_trainer import BaseTrainer
from geld_cvrptw.utils.device import move_items_to_device
from geld_cvrptw.utils.experiment_tracker import ExperimentTracker, should_log_batch
from geld_cvrptw.utils.metrics import AverageMeter


class Stage2Trainer(BaseTrainer):
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
        super().__init__(
            env_params,
            model_params,
            optimizer_params,
            trainer_params,
            run_type="train_stage_2",
            tracker=tracker,
        )

        # Continnue from stage 1 finish checkpoint
        checkpoint_path = (
            f"{trainer_params['pretrained_dir']}/checkpoint-{trainer_params['pretrained_epoch']}.pt"
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.solver = build_pipeline(self.trainer_params["pipeline"])

    def run_training(self):
        """Run stage 2 supervised training with curriculum learning."""
        self.env.set_device(self.device)

        # load the dataset with the small instanes  once 
        self.small_instances = self._load_small_samples()
        

        # Training with increasing n_customers (curriculum learning)
        n_customers_min = self.trainer_params["n_customers_min"]
        n_customers_max = self.trainer_params["n_customers_max"]
        self.logger.info(
            f"Starting curriculum learning from n_customers={n_customers_min} to {n_customers_max}"
        )

        for epoch in range(self.start_epoch, self.trainer_params["epochs"] + 1):
            n_customers = (
                n_customers_min
                + epoch * (n_customers_max - n_customers_min) // self.trainer_params["epochs"]
            )

            train_reference_length, train_predicted_length, train_loss, learning_rate = self._train_one_epoch(
                epoch, n_customers
            )
            self.log_metrics(
                epoch,
                train_reference_length,
                train_predicted_length,
                train_loss,
                learning_rate,
            )
            self.save_model_checkpoints_and_progress(epoch)

    def _load_small_samples(self):
        """Load and return the stage 1 training dataset with size-100 problems."""
        dataset = load_cvrptw_data_with_labels()

        coords = dataset["coords"].requires_grad_(False)
        demand = dataset["demand"].requires_grad_(False)
        tw_start = dataset["tw_start"].requires_grad_(False)
        tw_end = dataset["tw_end"].requires_grad_(False)
        service_time = dataset["service_time"].requires_grad_(False)
        tours = dataset["label_tours"].requires_grad_(False)
        costs = dataset["costs"].requires_grad_(False)

        return move_items_to_device(
            (coords, demand, tw_start, tw_end, service_time, tours, costs),
            self.device,
        )

    def _generate_large_samples_with_pseudolabels(self, epoch: int, n_customers: int):
        """Generate instances at the current size and pseudo-label them with beam search."""
        alpha = Uniform(0.5, 3.0).sample().item()
        num_samples = self.trainer_params["instances_per_epoch"]
        gen_batch_size = self.trainer_params["batch_size"]

        tensors = generate_instance_tensors(
            num_samples,
            problem_size=n_customers,
            alpha=alpha,
            seed=epoch,
            batch_size=gen_batch_size,
        )
        coords, demand, tw_start, tw_end, service_time = move_items_to_device(
            (
                tensors["coords"],
                tensors["demand"],
                tensors["tw_start"],
                tensors["tw_end"],
                tensors["service_time"],
            ),
            self.device,
        )

        self.env.set_batch(coords, demand, tw_start, tw_end, service_time)
        solve_result = self.solver.run(self.model, self.env)
        self.env.reset()

        batch_size = solve_result.tour.size(0)
        depot_column = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
        tour_rows = [
            torch.cat((depot_column[i], solve_result.tour[i]), dim=0)
            for i in range(batch_size)
        ]
        tours = torch.nn.utils.rnn.pad_sequence(
            tour_rows, batch_first=True, padding_value=TOUR_PAD_VALUE
        )
        costs = solve_result.length_normalized

        return coords, demand, tw_start, tw_end, service_time, tours, costs

    def _train_one_epoch(self, epoch: int, n_customers: int):
        """
        1. Generate data of the current size (mixed with size-100 problems).
        2. Pseudo-label large instances with beam search.
        3. Supervised training on alternating large/small batches.
        """
        large_instances = self._generate_large_samples_with_pseudolabels(epoch, n_customers)

        batch_size = self.trainer_params["batch_size"]
        large_batches = list(
            iter_tensor_batches(large_instances, batch_size=batch_size, shuffle=True)
        )
        num_batches = len(large_batches)
        small_batches = list(
            iter_tensor_batches(
                self.small_instances,
                batch_size=batch_size,
                shuffle=True,
                num_batches=num_batches,
            )
        )

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
                current_batch_size = current_batch[0].size(0)
                reference_length, predicted_length, avg_loss = self._train_one_batch(current_batch)

                # Logging & Tracking
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

    def _train_one_batch(self, batch):
        """Load one in-memory batch and run supervised training."""
        coords, demand, tw_start, tw_end, service_time, tours, costs = batch
        self.env.set_batch(
            coords,
            demand,
            tw_start,
            tw_end,
            service_time,
            label_tours=tours,
            label_costs=costs,
            train=True,
        )
        return self._train_supervised_batch(coords.size(0))
