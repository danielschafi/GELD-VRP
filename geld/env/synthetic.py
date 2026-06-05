"""Environment for synthetic and LEHD training data."""

import random
from logging import getLogger

import torch
from tqdm import tqdm

from geld.data.augmentations import apply_rotation, maybe_reverse_tour, sample_training_subpath
from geld.data.loaders import load_lehd_line, load_tsp_instances_with_baselines
from geld.env.base import TSPEnvironmentBase
from geld.paths import benchmark_data_dir

logger = getLogger(__name__)


class SyntheticEnvironment(TSPEnvironmentBase):
    """Environment for SL/SIL training on synthetic and LEHD TSP instances."""

    def __init__(self, **env_params):
        super().__init__(**env_params)
        self.raw_data_nodes = []
        self.raw_data_tours = []
        self.raw_data_nodes_100 = []
        self.raw_data_tours_100 = []

    def load_problems(self, batch_offset, batch_size, mix_curriculum_sizes=False, train=False):
        """Load a training batch with optional curriculum mixing and augmentation."""
        self.batch_offset = batch_offset
        self.batch_size = batch_size

        if mix_curriculum_sizes:
            index = random.sample(range(1_000_000), batch_size)
            problems_small = self.raw_data_nodes_100[index]
            solution_small = self.raw_data_tours_100[index]
            problems_large = self.raw_data_nodes[batch_offset : batch_offset + batch_size]
            solution_large = self.raw_data_tours[batch_offset : batch_offset + batch_size]
            if self.use_subpath_augmentation:
                problems_large, solution_large = sample_training_subpath(
                    problems_large, solution_large, mode="train", low_index=101
                )
            solution_large = maybe_reverse_tour(solution_large)
            self.problem_size = problems_large.shape[1]
            node_gap = self.problem_size - 100
            batch_indices = torch.arange(batch_size, dtype=torch.long, device=problems_large.device)
            anchor = problems_small[batch_indices, solution_small[:, 0]].unsqueeze(1).repeat(1, node_gap, 1)
            problems_small = torch.cat((anchor, problems_small), dim=1)
            prefix_indices = torch.arange(node_gap, dtype=torch.long, device=problems_large.device)[None, :].repeat(
                batch_size, 1
            )
            solution_small = torch.cat((prefix_indices, solution_small + node_gap), dim=1)
            self.problems = torch.cat((problems_small, problems_large), dim=0)
            self.label_tour = torch.cat((solution_small, solution_large), dim=0)
            self.batch_size = batch_size * 2
        else:
            self.problems = self.raw_data_nodes[batch_offset : batch_offset + batch_size]
            self.label_tour = self.raw_data_tours[batch_offset : batch_offset + batch_size]
            if self.use_subpath_augmentation:
                self.problems, self.label_tour = sample_training_subpath(
                    self.problems, self.label_tour, mode="train"
                )
            self.label_tour = maybe_reverse_tour(self.label_tour)
            self.problem_size = self.problems.shape[1]

        if train:
            rotation_id = torch.randint(low=0, high=8, size=[1])[0].item()
            self.problems = apply_rotation(self.problems, rotation_id)
        self.sync_batch_to_device()

    def shuffle_data(self):
        """Shuffle stored training instances."""
        index = torch.randperm(len(self.raw_data_nodes)).long()
        self.raw_data_nodes = self.raw_data_nodes[index]
        self.raw_data_tours = self.raw_data_tours[index]

    def generate_random_instances(self, batch_size, problem_size):
        """Generate random uniform TSP instances for stage-2 SIL."""
        self.batch_size = batch_size
        self.problem_size = problem_size
        self.raw_data_nodes = torch.rand(
            size=(batch_size, problem_size, 2), device=self.device, requires_grad=False
        )
        self.raw_data_tours = None

    def load_problems_val(self, batch_offset, batch_size, rotation_id=0):
        """Load validation coordinates with optional ×8 data augmentation."""
        self.problems = self.raw_data_nodes[batch_offset : batch_offset + batch_size]
        if rotation_id != 0:
            self.problems = apply_rotation(self.problems, rotation_id)
        self.label_tour = None

    def load_raw_data(
        self,
        num_instances,
        begin_index=0,
        load_eval_data=True,
        load_synthetic_benchmark=False,
        size=None,
        distribution=None,
    ):
        """Load LEHD training data, curriculum subset, or synthetic benchmarks."""
        logger.info("Loading raw dataset...")
        if load_eval_data:
            if load_synthetic_benchmark:
                root = benchmark_data_dir()
                instances, baseline_tours, _ = load_tsp_instances_with_baselines(root, size, distribution)
                self.raw_data_nodes = instances
                self.raw_data_tours = baseline_tours
            else:
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
        else:
            nodes_list = []
            tours_list = []
            with open(self.data_path, "r", encoding="utf-8") as data_file:
                lines = data_file.readlines()[begin_index : num_instances + begin_index]
            for line in tqdm(lines, ascii=True):
                nodes, tour = load_lehd_line(line)
                nodes_list.append(nodes)
                tours_list.append(tour)
            self.raw_data_nodes_100 = torch.tensor(nodes_list, requires_grad=False)
            self.raw_data_tours_100 = torch.tensor(tours_list, requires_grad=False)
            logger.info("Raw 100-node curriculum dataset loaded.")
