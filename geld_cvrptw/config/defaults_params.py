from geld_cvrptw.config.paths import project_root, training_stage_1_data_dir


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


def default_decoder_config(
    *,
    name: str = "beam_search",
    beam_size: int = 16,
    horizon_factor: int = 4,
) -> dict:
    """Shared decoder configuration for eval, scaling, and stage-2 training."""
    return {
        "name": name,
        "beam_size": beam_size,
        "horizon_factor": horizon_factor,
    }


def default_reconstruction_config(
    *,
    enabled: bool = False,
    rc_iterations: int = 100,
    window_size_min: int = 4,
    num_windows_min: int = 2,
    augment_coords: bool = False,
) -> dict:
    """Shared reconstruction post-processor configuration."""
    return {
        "enabled": enabled,
        "rc_iterations": rc_iterations,
        "window_size_min": window_size_min,
        "num_windows_min": num_windows_min,
        "augment_coords": augment_coords,
    }


def default_pipeline_config(
    *,
    beam_size: int = 16,
    reconstruction_enabled: bool = False,
) -> dict:
    """Pipeline config for build_pipeline (decoder + optional reconstruction)."""
    return {
        "decoder": default_decoder_config(beam_size=beam_size),
        "reconstruction": default_reconstruction_config(enabled=reconstruction_enabled),
    }


def default_training_stage_1_optimizer_params() -> dict:
    """Stage-1 Adam optimizer and MultiStepLR scheduler (lr=1e-4)."""
    return {
        "optimizer": {"lr": 1e-4},
        "scheduler": {
            "milestones": [20, 35, 45],
            "gamma": 0.5,
        },
    }


def default_training_stage_2_optimizer_params() -> dict:
    """Stage-2 SIL optimizer (lr=1e-5)."""
    return {
        "optimizer": {"lr": 1e-5},
        "scheduler": {
            "milestones": [20, 35, 45],
            "gamma": 0.5,
        },
    }


def default_training_stage_1_params(use_cuda: bool = True, cuda_device_num: int = 0) -> dict:
    """Stage-1 SL training defaults (50 epochs, batch 1024)."""
    return {
        "use_cuda": use_cuda,
        "cuda_device_num": cuda_device_num,
        "epochs": 50,
        "instances_per_epoch": 1_000_000,
        "batch_size": 1024,
        "logging": {
            "model_save_interval": 1,
            "batch_log_interval": 50,
        },
        "resume_checkpoint": {
            "enable": False,
            "path": str(project_root() / "result" / "None"),
            "epoch": 1,
        },
    }


def default_training_stage_2_params(
    use_cuda: bool = True,
    cuda_device_num: int = 0,
    pretrained_dir: str | None = None,
    pretrained_epoch: int = 1,
) -> dict:
    """Stage-2 SIL curriculum defaults (n_customers 100→1000, beam width 16)."""
    if pretrained_dir is None:
        pretrained_dir = str(project_root() / "result" / "Here")
    params = default_training_stage_1_params(use_cuda, cuda_device_num)
    beam_size = 16
    params.update(
        {
            "instances_per_epoch": 512,
            "batch_size": 64,
            "n_customers_min": 100,
            "n_customers_max": 1000,
            "pretrained_dir": pretrained_dir,
            "pretrained_epoch": pretrained_epoch,
            "pipeline": default_pipeline_config(beam_size=beam_size, reconstruction_enabled=False),
        }
    )
    return params


def default_cvrptw_env_params() -> dict:
    """Minimal env params for CVRPTW inference."""
    return {}


def default_cvrptw_eval_params(use_cuda: bool = True, cuda_device_num: int = 0) -> dict:
    """CVRPTW evaluation defaults (beam search decoder, optional reconstruction)."""
    return {
        "use_cuda": use_cuda,
        "cuda_device_num": cuda_device_num,
        "model_load": {
            "path": str(project_root() / "result" / "pre_trained_model"),
            "epoch": 49,
        },
        "synthetic": {
            "n_customers": 100,
            "num_instances": 1000,
            "batch_size": 100,
        },
        "decoder": default_decoder_config(),
        "reconstruction": default_reconstruction_config(enabled=True),
    }


def default_scaling_benchmark_params(use_cuda: bool = True, cuda_device_num: int = 0) -> dict:
    """Defaults for synthetic beam-search scaling benchmark (decode-only, no RC)."""
    return {
        "use_cuda": use_cuda,
        "cuda_device_num": cuda_device_num,
        "model_load": {
            "path": str(project_root() / "result" / "pre_trained_model"),
            "epoch": 49,
        },
        "n_customers_values": [100, 200, 500, 1000, 2000, 5000],
        "num_instances": None,
        "decode_batch_size": None,
        "seed": 2024,
        "alpha": 1.0,
        "decoder": default_decoder_config(),
        "reconstruction": default_reconstruction_config(enabled=False),
    }
