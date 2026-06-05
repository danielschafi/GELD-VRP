"""Environment for TSPLIB and National TSP benchmarks."""

from logging import getLogger

import torch
from tqdm import tqdm

from geld.data.augmentations import maybe_reverse_tour, sample_training_subpath
from geld.data.loaders import load_lehd_line, load_tsplib_file
from geld.env.base import TSPEnvironmentBase
from geld.paths import benchmark_data_dir

logger = getLogger(__name__)


class TSPLIBEnvironment(TSPEnvironmentBase):
    """Environment for TSPLIB and National TSP benchmark evaluation."""

    def load_problems(self, batch_offset, batch_size, name=None, opt_len=None, use_tsplib_dir=False, **_kwargs):
        """Load LEHD batch or a single real-world TSP instance."""
        self.batch_offset = batch_offset
        self.batch_size = batch_size
        if not self.eval_tsplib:
            self.problems = self.raw_data_nodes[batch_offset : batch_offset + batch_size]
            self.label_tour = self.raw_data_tours[batch_offset : batch_offset + batch_size]
            if self.use_subpath_augmentation:
                self.problems, self.label_tour = sample_training_subpath(
                    self.problems, self.label_tour, mode="train"
                )
            self.label_tour = maybe_reverse_tour(self.label_tour)
        else:
            self.tsplib_name = name
            self.tsplib_cost = opt_len
            instance, _ = load_tsplib_file(
                root=benchmark_data_dir(), tsplib_name=name, use_tsplib_dir=use_tsplib_dir
            )
            self.problems = instance.reshape(1, -1, 2)
            self.label_tour = None
        self.problem_size = self.problems.shape[1]
        self.sync_batch_to_device()

    def load_raw_data(self, num_instances, begin_index=0):
        """Load LEHD training instances into raw tensors."""
        logger.info("Loading raw dataset...")
        nodes_list = []
        tours_list = []
        with open(self.data_path, "r", encoding="utf-8") as data_file:
            lines = data_file.readlines()[begin_index : num_instances + begin_index]
        for line in tqdm(lines, ascii=True):
            nodes, tour = load_lehd_line(line)
            nodes_list.append(nodes)
            tours_list.append(tour)
        self.raw_data_nodes = torch.tensor(nodes_list, requires_grad=False)
        self.raw_data_tours = torch.tensor(tours_list, requires_grad=False)
        logger.info("Raw dataset loaded.")
