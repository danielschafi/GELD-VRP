"""Experiment logging and result directory management."""

import logging
import logging.config
import sys
from datetime import datetime
from pathlib import Path

import pytz

from geld_cvrptw.config.paths import project_root
from geld_cvrptw.utils.metrics import LogData

_result_folder: Path | None = None


def get_result_folder() -> Path:
    """Return timestamped result directory, creating it on first access."""
    global _result_folder
    if _result_folder is None:
        process_start_time = datetime.now(pytz.timezone("Europe/Zurich"))
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
