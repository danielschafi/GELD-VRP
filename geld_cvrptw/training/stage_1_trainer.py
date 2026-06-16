"""
class that runs the stage 1 training procedure with supervised learning labels.
"""

from logging import getLogger

import torch
from torch.optim import Adam 
from torch.optim.lr_scheduler import MultiStepLR 

from geld_cvrptw.utils.experiment_tracker import ExperimentTracker
from geld_cvrptw.utils.metrics import AverageMeter, LogData, TimeEstimator
from geld_cvrptw.utils.logging import (
    get_result_folder,
    util_print_log_array,
    util_save_log_image_with_label,
)
from geld.utils.device import setup_device


from geld_cvrptw.model.GELD_CVRPTW import GeldCvrptwModel
from geld_cvrptw.env.CVRPTW import CVRPTWEnv

class Stage1Trainer:
    """ Stage 1 trainer. Trains on small instances (100 customers)"""
    def __init__(self, 
        env_params:dict,
        model_params:dict,
        optimizer_params:dict,
        trainer_params:dict,
        tracker:ExperimentTracker | None = None):

        # Setup logging
        self.logger = getLogger(name="trainer")
        self.result_folder = get_result_folder()
        self.result_log = LogData()
        self.tracker = tracker

        # Where to run training
        self.device = setup_device(
            trainer_params["use_cuda"], trainer_params["cuda_device_num"]
        )
        """Initializes model etc with hyperparams from config. """
        # Reproducability
        torch.manual_seed(42)

        # Save Params
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # Initialize model, env, optim + scheduler and set hyperparams
        self.model = GeldCvrptwModel(**self.model_params).to(self.device)  # TODO: implemenmt
        self.env = CVRPTWEnv(**self.env_params) # TODO: Implement
        self.env.set_device(self.device) # TODO: Implement

        self.optimizer = Adam(
            self.model.parameters(), **self.optimizer_params["optimizer"]
        )
        self.scheduler = MultiStepLR(self.optimizer, **self.optimizer_params["scheduler"])

        # Training restart / continue handling
        self.start_epoch = 1
        if self.trainer_params.get("model_load", {}).get("enable", False):
            self.continue_training_from_checkpoint()

        # Training time estimator
        self.time_estimator = TimeEstimator()


    def continue_training_from_checkpoint(self):
        """
        continues training from a checkpoint according to config. 
        Loads params and model, optim etc. state
        """
        model_load = self.trainer_params["model_load"]
        checkpoint_path = (
            f"{model_load['path']}/checkpoint-{model_load['epoch']}.pt"
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.start_epoch = 1 + model_load["epoch"]
        self.result_log.set_raw_data(checkpoint["result_log"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.last_epoch = model_load["epoch"] - 1
        self.logger.info("Saved Model Loaded !!")


    def run(self):
        """Run stage 1 supervised training"""

        # Load Train Data into memory.
        self.env.load_raw_data(self.trainer_params["train_episodes"])
        self.env.set_device(self.device)

        for epoch in range(self.start_epoch, self.trainer_params["epochs"] + 1):

            self.env.shuffle_data()
            train_reference_length, train_predicted_length, train_loss = self._train_one_epoch(epoch)
            self.scheduler.step()

            self.log_metrics(epoch, train_reference_length, train_predicted_length, train_loss)
            self.log_training_curve(epoch)
            self.save_model_checkpoints_and_progress(epoch)



    def log_metrics(self, epoch:int, train_reference_length:float, train_predicted_length:float, train_loss:float):
        """Does all the logging, tracking to wandb etc for one epoch"""
        self.logger.info(
            "================================================================="
        )

        # Log
        self.result_log.append(
            "train_reference_length", epoch, train_reference_length
        )
        self.result_log.append(
            "train_predicted_length", epoch, train_predicted_length
        )
        self.result_log.append("train_loss", epoch, train_loss)

        elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(
            epoch, self.trainer_params["epochs"]
        )
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
    
    def log_training_curve(self,epoch:int):
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


    def save_model_checkpoints_and_progress(self, epoch:int):
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
