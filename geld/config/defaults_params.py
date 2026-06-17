"""Default hyperparameters matching the original GELD scripts."""

from geld.paths import project_root, training_stage_1_data_dir


def default_model_params(mode: str = "train") -> dict:
    """Default GE/LD hyperparameters (h=128, 6 LD layers, 8 heads)."""
    return {
        "mode": mode,
        "embedding_dim": 128,
        "decoder_layer_num": 6,
        "qkv_dim": 16,
        "head_num": 8,
        "ff_hidden_dim": 128,
    }


def default_env_params(mode: str = "train", use_subpath_augmentation: bool = True) -> dict:
    """Default environment settings with LEHD TSP-100 training path."""
    training_path = training_stage_1_data_dir() / "train_TSP100_n100w-001.txt"
    return {
        "data_path": str(training_path),
        "mode": mode,
        "use_subpath_augmentation": use_subpath_augmentation,
        "eval_tsplib": False,
    }


def default_training_stage_1_optimizer_params() -> dict:
    """Stage-1 Adam optimizer and MultiStepLR scheduler (lr=1e-4)."""
    return {
        "optimizer": {"lr": 1e-4},
        "scheduler": {
            "milestones": [i for i in range(1, 50)],
            "gamma": 0.97,
        },
    }


def default_training_stage_2_optimizer_params() -> dict:
    """Stage-2 SIL optimizer (lr=1e-5)."""
    return {"optimizer": {"lr": 1e-5}}


def default_training_stage_1_params(use_cuda: bool = True, cuda_device_num: int = 0) -> dict:
    """Stage-1 SL training defaults (n_e1=50 epochs, batch 1024)."""
    return {
        "use_cuda": use_cuda,
        "cuda_device_num": cuda_device_num,
        "epochs": 50,
        "train_episodes": 1_000_000,
        "train_batch_size": 1024,
        "logging": {
            "model_save_interval": 1,
            "img_save_interval": 3000,
            "batch_log_interval": 50,
            "log_image_params_1": {
                "json_foldername": "log_image_style",
                "filename": "style_tsp_100.json",
            },
            "log_image_params_2": {
                "json_foldername": "log_image_style",
                "filename": "style_loss_1.json",
            },
        },
        "model_load": {
            "enable": False,
            "path": str(project_root() / "result" / "None"),
            "epoch": 1,
        },
    }


def default_training_stage_2_params(
    use_cuda: bool = True,
    cuda_device_num: int = 0,
    model_load_path: str | None = None,
    model_load_epoch: int = 1,
) -> dict:
    """Stage-2 SIL curriculum defaults (k_m=100, n_max=1000, BS width 16)."""
    if model_load_path is None:
        model_load_path = str(project_root() / "result" / "Here")
    params = default_training_stage_1_params(use_cuda, cuda_device_num)
    params.update(
        {
            "train_episodes": 512,
            "train_batch_size": 64,
            "val_batch_size": 512,
            "val_beam_batch_size": 512,
            "beam_size": 16,
            "max_limit": 5,
            "per_batch": 5,
            "best_limit": 3,
            "problem_size_init": 100,
            "problem_size_max": 1000,
            "model_load_path": model_load_path,
            "model_load_epoch": model_load_epoch,
        }
    )
    return params


def default_eval_params(use_cuda: bool = True, cuda_device_num: int = 0) -> dict:
    """Evaluation defaults with BS (B=16) and PRC (1000 iterations)."""
    return {
        "use_cuda": use_cuda,
        "cuda_device_num": cuda_device_num,
        "test_episodes": 200,
        "test_batch_size": 200,
        "beam_size": 16,
        "num_PRC": 1000,
        "beam": True,
        "PRC": True,
        "model_load": {
            "path": str(project_root() / "result" / "pre_trained_model"),
            "epoch": 49,
        },
    }
