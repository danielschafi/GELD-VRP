"""Beam search over decoder transition probabilities."""

import torch


class BeamSearch:
    """Beam search (BS) over per-step transition log-probabilities."""

    def __init__(
        self,
        beam_size,
        batch_size,
        num_nodes,
        dtype_float=torch.FloatTensor,
        dtype_long=torch.LongTensor,
        probs_type="raw",
        random_start=False,
        device=torch.device("cpu"),
    ):
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.num_nodes = num_nodes
        self.probs_type = probs_type
        self.dtype_float = dtype_float
        self.dtype_long = dtype_long
        self.device = device

        self.start_nodes = torch.zeros(batch_size, beam_size, device=device).type(
            dtype_long
        )
        if random_start:
            self.start_nodes = torch.randint(
                0, num_nodes, (batch_size, beam_size), device=device
            ).type(dtype_long)

        self.mask = torch.ones(batch_size, beam_size, num_nodes, device=device).type(
            dtype_float
        )
        self.update_mask(self.start_nodes)
        self.scores = torch.zeros(batch_size, beam_size, device=device).type(
            dtype_float
        )
        self.all_scores = []
        self.parent_beam_indices = []
        self.next_nodes = [self.start_nodes]

    @torch.no_grad()
    def advance(self, trans_probs, selected_nodes):
        """Expand beam, retain top-B sub-tours, and update visit mask."""
        if len(self.parent_beam_indices) > 0:
            if self.probs_type == "raw":
                beam_scores = trans_probs * self.scores.unsqueeze(2).expand_as(
                    trans_probs
                )
            else:
                beam_scores = trans_probs + self.scores.unsqueeze(2).expand_as(
                    trans_probs
                )
        else:
            beam_scores = trans_probs
            if self.probs_type == "raw":
                beam_scores[:, 1:] = torch.zeros(
                    beam_scores[:, 1:].size(), device=self.device
                ).type(self.dtype_float)
            else:
                beam_scores[:, 1:] = -1e20 * torch.ones(
                    beam_scores[:, 1:].size(), device=self.device
                ).type(self.dtype_float)

        beam_scores = beam_scores * self.mask
        beam_scores = beam_scores.view(self.batch_size, -1)
        best_scores, best_scores_id = beam_scores.topk(self.beam_size, 1, True, True)
        self.scores = best_scores

        parent_beam_idx = torch.div(
            best_scores_id, self.num_nodes, rounding_mode="trunc"
        )
        self.parent_beam_indices.append(parent_beam_idx)
        new_nodes = best_scores_id - parent_beam_idx * self.num_nodes
        self.next_nodes.append(new_nodes)

        perm_mask = parent_beam_idx.unsqueeze(2).expand_as(self.mask)
        self.mask = self.mask.gather(1, perm_mask)

        selected_nodes = selected_nodes.view(self.batch_size, self.beam_size, -1)
        perm_selected_nodes = parent_beam_idx.unsqueeze(2).expand_as(selected_nodes)
        selected_nodes = selected_nodes.gather(1, perm_selected_nodes)
        selected_nodes = selected_nodes.view(self.batch_size * self.beam_size, -1)
        self.update_mask(new_nodes)
        return selected_nodes

    def update_mask(self, new_nodes):
        """Mask visited nodes in the current beam."""
        node_indices = (
            torch.arange(0, self.num_nodes, device=self.device)
            .unsqueeze(0)
            .unsqueeze(1)
            .expand_as(self.mask)
            .type(self.dtype_long)
        )
        new_nodes = new_nodes.unsqueeze(2).expand_as(self.mask)
        visit_mask = 1 - torch.eq(node_indices, new_nodes).type(self.dtype_float)
        self.mask = self.mask * visit_mask
        if self.probs_type == "logits":
            self.mask[self.mask == 0] = 1e20
