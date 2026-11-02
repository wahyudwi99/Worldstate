import torch

from world_models.rollout import imagine, warm_state


class IncrementDynamics(torch.nn.Module):
    """Known dynamics lets the test detect observed-latent teacher forcing."""
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, z, actions, state=None):
        self.inputs.append(z.clone())
        means = (z + actions[..., :1] + 1).unsqueeze(-1)
        return (torch.zeros_like(means), means, torch.zeros_like(means)), state


def test_free_running_feeds_generated_latents():
    model = IncrementDynamics()
    result = imagine(model, torch.zeros(1, 2), torch.zeros(1, 3, 3), deterministic=True)
    torch.testing.assert_close(result, torch.tensor([[[1., 1.], [2., 2.], [3., 3.]]]))
    assert len(model.inputs) == 3
    torch.testing.assert_close(model.inputs[1], result[:, :1])


def test_empty_history_and_reset():
    model = IncrementDynamics()
    assert warm_state(model, torch.empty(1, 0, 2), torch.empty(1, 0, 3)) is None
    a = imagine(model, torch.zeros(1, 2), torch.zeros(1, 2, 3), deterministic=True)
    b = imagine(model, torch.zeros(1, 2), torch.zeros(1, 2, 3), deterministic=True)
    torch.testing.assert_close(a, b)
