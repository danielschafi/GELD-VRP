import torch

from geld.search.beam_search import BeamSearch


def test_beam_search_picks_best_path():
    batch_size = 1
    num_nodes = 3
    beam_size = 2
    device = torch.device("cpu")
    beam = BeamSearch(
        beam_size,
        batch_size,
        num_nodes,
        torch.FloatTensor,
        torch.LongTensor,
        probs_type="logits",
        device=device,
    )
    # Step 1 logits favor node 1 strongly on first beam, node 2 on exploration
    logits = torch.zeros(batch_size, beam_size, num_nodes)
    logits[0, 0, 1] = 0.0
    logits[0, 0, 2] = -10.0
    logits[0, 1, 2] = 0.0
    logits[0, 1, 1] = -10.0
    selected = torch.zeros(batch_size * beam_size, 0, dtype=torch.long)
    updated = beam.advance(logits, selected)
    assert updated.shape == (batch_size * beam_size, 0)
