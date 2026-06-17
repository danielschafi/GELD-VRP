"""Experiment logging and result directory management."""

import json
import logging
import logging.config
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pytz

from geld.paths import project_root
from geld.utils.metrics import LogData

_result_folder: Path | None = None


def get_result_folder() -> Path:
    """Return timestamped result directory, creating it on first access."""
    global _result_folder
    if _result_folder is None:
        process_start_time = datetime.now(pytz.timezone("Asia/Seoul"))
        _result_folder = project_root() / "result" / process_start_time.strftime("%Y%m%d_%H%M%S")
    return _result_folder


def set_result_folder(folder: Path | str):
    """Override the global result directory path."""
    global _result_folder
    _result_folder = Path(folder)


def create_logger(log_file: dict | None = None, **kwargs):
    """Configure root logger and result directory."""
    if log_file is None:
        log_file = kwargs.get("log_file", {})
    filepath = log_file.get("filepath", get_result_folder())
    if "desc" in log_file:
        filepath = Path(str(filepath).format(desc="_" + log_file["desc"]))
    else:
        filepath = Path(str(filepath).format(desc=""))
    set_result_folder(filepath)
    filename = filepath / log_file.get("filename", "log.txt")
    filename.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    file_mode = "a" if filename.is_file() else "w"
    file_handler = logging.FileHandler(filename, mode=file_mode)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)


def util_print_log_array(logger, result_log: LogData):
    """Log all metric series from training/eval history."""
    for key in result_log.get_keys():
        logger.info(f"{key}_list = {result_log.get(key)}")


def util_save_log_image_with_label(result_file_prefix, img_params, result_log: LogData, labels=None):
    """Save a training curve plot as JPG."""
    dirname = os.path.dirname(result_file_prefix)
    os.makedirs(dirname, exist_ok=True)
    _build_log_image_plt(img_params, result_log, labels)
    if labels is None:
        labels = result_log.get_keys()
    file_name = "_".join(labels)
    fig = plt.gcf()
    fig.savefig(f"{result_file_prefix}-{file_name}.jpg")
    plt.close(fig)


def _build_log_image_plt(img_params, result_log: LogData, labels=None):
    """Build matplotlib figure from JSON style config and log data."""
    style_dir = Path(__file__).resolve().parent / "log_image_style"
    config_path = style_dir / img_params["filename"]
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    figsize = (config["figsize"]["x"], config["figsize"]["y"])
    plt.figure(figsize=figsize)
    if labels is None:
        labels = result_log.get_keys()
    for label in labels:
        plt.plot(*result_log.getXY(label), label=label)

    ylim_min = config["ylim"]["min"]
    ylim_max = config["ylim"]["max"]
    if ylim_min is None:
        ylim_min = plt.gca().dataLim.ymin
    if ylim_max is None:
        ylim_max = plt.gca().dataLim.ymax
    plt.ylim(ylim_min, ylim_max)

    xlim_min = config["xlim"]["min"]
    xlim_max = config["xlim"]["max"]
    if xlim_min is None:
        xlim_min = plt.gca().dataLim.xmin
    if xlim_max is None:
        xlim_max = plt.gca().dataLim.xmax
    plt.xlim(xlim_min, xlim_max)
    plt.rc("legend", **{"fontsize": 18})
    plt.legend()
    plt.grid(config["grid"])
