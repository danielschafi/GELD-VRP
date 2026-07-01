"""Re-Construction (RC) post-processor for CVRPTW."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from geld_cvrptw.data.augmentations import apply_random_rotation
from geld_cvrptw.data.loaders import TOUR_PAD_VALUE
from geld_cvrptw.env.CVRPTW import CVRPTWEnv, DynamicState, StaticState
from geld_cvrptw.inference.types import PostProcessor, SolveResult
from geld_cvrptw.model.GELD_CVRPTW import GeldCvrptwModel
from geld_cvrptw.model.helpers import LARGE_INSTANCE_THRESHOLD

@dataclass
class RollingVehicleState:
    """Records the remaining capacity and time at each node throughout the tour."""
    remaining_capacity:torch.Tensor
    current_time:torch.Tensor

class ReConstruction(PostProcessor):
    """
    Re-Construction post-processor for CVRPTW.
    """

    def __init__(
        self,
        num_iterations: int = 1000,
        min_window_length: int = 4,
        min_window_count: int = 2,
        diversify_coords: bool = False,
    ) -> None:
        self.num_iterations = num_iterations
        #self.min_window_length = min_window_length
        #self.min_window_count = min_window_count
        #self.diversify_coords = diversify_coords
        self.segment_len_min = 4
        self.batch_size = None
        self.problem_size = None
        self.device = None

    @torch.no_grad()
    def refine(self, model: GeldCvrptwModel, env: CVRPTWEnv, initial_result: SolveResult) -> SolveResult:
        """Run RC for ``num_iterations`` and return the best refined tour."""
        tour = initial_result.tour.clone()
        self.batch_size = tour.size(0)
        self.problem_size = tour.size(1) 
        self.device = env.device 
        
        static_state = env._build_static_state()
        model.embed_static_state_once(static_state)
        rolling_state = self._compute_rolling_vehicle_state(env, tour)

        # Segment size bounds
        # TODO: Smaller better ? 
        large_instance = self.problem_size > LARGE_INSTANCE_THRESHOLD
        self.segment_len_max = self.problem_size // (4 if not large_instance else 10)

        for _ in range(self.num_iterations):
            tour, rolling_state = self.run_one_iteration(model, env, tour, rolling_state)

        length = env.compute_tour_length(env.batch_coords, tour)
        return SolveResult(tour=tour, length_normalized=length)

    @torch.no_grad()
    def run_one_iteration(
        self,
        model: GeldCvrptwModel,
        env: CVRPTWEnv,
        tour: torch.Tensor,
        schedule: RollingVehicleState,
    ) -> tuple[torch.Tensor, RollingVehicleState]:
        """
        One RC iteration:

        1. Optionally rotate coordinates (off by default — each rotation re-embeds)
        2. Plan evenly spaced repair windows
        3. Try improving every window in parallel
        4. Merge windows that are shorter and still yield a feasible full tour
        """

        # Set up segments
        num_segments, segment_boundary_indices, segment_len = self._get_segment_boundary_indices()
        sub_problems, label_tours_in_canonical_order = self._batch_segments(
            env.batch_coords, tour, segment_boundary_indices, segment_len
        )
        # Augmentation
        origin_coords = env.batch_coords 
        rotation_id = torch.randint(0,1, size=[1], device=self.device)[0].item()
        
        # Re-embedd nodes if they have been changed (apply encoding + RALA): coords changed everything else stayed the same
        if rotation_id != 0:
            env.batch_coords = apply_random_rotation(origin_coords, rotation_id)
            static_state = env._build_static_state()
            model.embed_static_state_once(static_state)






    def improve_segments(self, segments, model:GeldCvrptwModel, env:CVRPTWEnv):
        pass



    def _batch_segments(
        self,
        node_features: torch.Tensor,
        tour: torch.Tensor,
        segment_boundary_indices: torch.Tensor,
        segment_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract and canonicalize tour segments for parallel RC.

        Parameters:
            node_features: (B, N, F) per-node features (e.g. coords; F=2 for (x, y) only).
            tour: (B, N) full tour per instance, node IDs in visit order.
            segment_boundary_indices: (S,) start positions of S evenly spaced segments.
            segment_len: (L) consecutive tour positions per segment; also stride between starts.

        Returns:
            segment_node_features: (B * S, L, F) features per segment, nodes sorted by ID
                within each segment (canonical sub-problem order).
            label_tours_in_canonical_order: (B * S, L) visit order as indices into that
                canonical order (slots 0..L-1, not original node IDs).
        """

        # extract segments of length segment_len for each batch. 
        positions_within_segment = torch.arange(segment_len, device=tour.device)
        segment_tour_positions = segment_boundary_indices.unsqueeze(1) + positions_within_segment # (num_segments, segment_len)
        node_ids_in_visit_order = tour[:, segment_tour_positions] #  (batch, num_segments, segment_len)
        
        # each segments node ids sorted 
        # [12,4,8,20] -> [1,2,0,3]
        node_id_sort_indices = node_ids_in_visit_order.argsort(dim=-1) # (batch, num_segments, segment_len)
        # [12,4,8,20] -> [4,8,12,20]
        canonical_node_ids = node_ids_in_visit_order.gather(dim=-1, index=node_id_sort_indices) # instead of running sort
        
        # [1,2,0,3] -> [2,0,1,3]
        visit_order_in_canonical_slots = node_id_sort_indices.argsort(dim=-1)

        batch_size = tour.size(0)
        num_segments = node_ids_in_visit_order.size(1)
        flat_canonical_node_ids = canonical_node_ids.reshape(batch_size, -1)
        instance_indices = torch.arange(batch_size, device=tour.device)[:, None].expand_as(
            flat_canonical_node_ids
        )

        segment_node_features = node_features[instance_indices, flat_canonical_node_ids.long()].view(
            batch_size * num_segments, segment_len, node_features.size(-1)
        )
        label_tours_in_canonical_order = visit_order_in_canonical_slots.reshape(
            batch_size * num_segments, segment_len
        )
        return segment_node_features, label_tours_in_canonical_order

    def _get_segment_boundary_indices(self) -> tuple[int, torch.Tensor, int]:
        """
        Generates the segment boundary indices for the tour.
        Appends shorter segments at the beginning and end (caused by the shift of the tour)

        Returns:
        - num_segments: int shape: (1)
        - segment_boundary_indices: torch.Tensor shape: (num_segments)
        - segment_len: int shape: (1)
        """
        num_segments = torch.randint(self.segment_len_min, self.segment_len_max+1, size=[1])[0]
        segment_len = self.problem_size // num_segments
        first_segment_offset = torch.randint(self.segment_len_min, segment_len, size=[1])[0]
        segment_boundary_indices = torch.arange(first_segment_offset, self.problem_size, step=segment_len, dtype=torch.long)


        # TODO: first and last would need padding i think
        # Add shorted segment at the beginning if it is not already there
        # if 0 != segment_boundary_indices[0]:
        #     segment_boundary_indices = torch.cat([torch.tensor([0]), segment_boundary_indices])
        
        #  # Add shorted segment at the end if it is not already there
        # if segment_boundary_indices[-1] != self.problem_size:
        #     segment_boundary_indices = torch.cat([segment_boundary_indices, torch.tensor([self.problem_size])])
        
        # Drop segments that extend past tour length
        if segment_boundary_indices[-1] + segment_len > self.problem_size:
            segment_boundary_indices = segment_boundary_indices[:-1] # drop last

        return num_segments, segment_boundary_indices, segment_len
        



    def _compute_rolling_vehicle_state(self, env: CVRPTWEnv, tour: torch.Tensor) -> RollingVehicleState:
        """Goes throught the tour and records the vehicles state at each node."""
        tour_lengths = self._num_steps_in_tour(tour)
        current_time, remaining_capacity = env.replay_tour_schedule(tour, tour_lengths)
        return RollingVehicleState(remaining_capacity=remaining_capacity, current_time=current_time)

    
    def _num_steps_in_tour(self, tour:torch.Tensor) -> torch.Tensor:
        """Returns how many steps were taken until all nodes were serviced (varies because of different nr of depot returns)"""
        # TODO: maybe move to env or utils
        return (tour != TOUR_PAD_VALUE).sum(dim=1)
    
