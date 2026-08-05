import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.model import ELF
from utils.sampling_utils import (
    _ode_step,
    _wff_ode_step,
    add_noise,
    make_wff_time_vector,
    net_out_to_v_x,
    sample_wff_timesteps,
)


class _Config:
    t_eps = 0.05
    self_cond_prob = 0.0
    num_self_cond_cfg_tokens = 0
    denoiser_noise_scale = 2.0


def _small_model():
    torch.manual_seed(7)
    return ELF(
        text_encoder_dim=8,
        max_length=6,
        hidden_size=32,
        depth=2,
        num_heads=4,
        bottleneck_dim=4,
        num_time_tokens=1,
        num_self_cond_cfg_tokens=0,
        num_model_mode_tokens=0,
        vocab_size=16,
        per_token_time_conditioning=True,
    ).eval()


class WFFTest(unittest.TestCase):
    def test_homogeneous_vector_time_matches_scalar_checkpoint_path(self):
        model = _small_model()
        x = torch.randn(2, 6, 8)
        scalar_t = torch.tensor([0.2, 0.7])
        vector_t = scalar_t[:, None].expand(-1, 6)
        scalar_out = model(x, scalar_t)[0]
        vector_out = model(x, vector_t)[0]
        torch.testing.assert_close(scalar_out, vector_out, rtol=0.0, atol=0.0)

    def test_local_time_gate_receives_gradient(self):
        model = _small_model().train()
        x = torch.randn(2, 6, 8)
        tau = torch.tensor([
            [0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ])
        model(x, tau)[0].square().mean().backward()
        self.assertIsNotNone(model.local_time_gate.grad)
        self.assertGreater(abs(float(model.local_time_gate.grad)), 0.0)

    def test_per_token_flow_algebra(self):
        x0 = torch.randn(2, 6, 8)
        noise = torch.randn_like(x0)
        tau = torch.linspace(0.1, 0.9, 6)[None].expand(2, -1)
        z = add_noise(x0, noise, tau, _Config())
        expected = tau[..., None] * x0 + (1.0 - tau[..., None]) * noise * 2.0
        torch.testing.assert_close(z, expected)
        v, x_recovered = net_out_to_v_x(x0, z, tau, t_eps=0.05)
        torch.testing.assert_close(x_recovered, x0)
        torch.testing.assert_close(v, (x0 - z) / (1.0 - tau[..., None]))

    def test_training_sampler_mixes_sync_and_wave_examples(self):
        torch.manual_seed(3)
        base = torch.full((128,), 0.5)
        tau, use_wave, delta, order = sample_wff_timesteps(
            base, 16, probability=0.5, delta_min=0.1, delta_max=0.2
        )
        self.assertEqual(tau.shape, (128, 16))
        self.assertTrue(bool(((use_wave == 0) | (use_wave == 1)).all()))
        self.assertGreater(float(use_wave.mean()), 0.25)
        self.assertLess(float(use_wave.mean()), 0.75)
        torch.testing.assert_close(
            tau[use_wave == 0], base[use_wave == 0][:, None].expand(-1, 16)
        )
        self.assertTrue(bool((delta[use_wave == 0] == 0).all()))
        self.assertTrue(bool(((order >= 0) & (order <= 2)).all()))

    def test_sampling_clock_has_synchronous_endpoints_and_is_monotone(self):
        grid = torch.linspace(0.0, 1.0, 101)
        clocks = torch.stack([
            make_wff_time_vector(
                float(s), 32, 0.2, "ltr", device=torch.device("cpu"), dtype=torch.float32
            )
            for s in grid
        ])
        torch.testing.assert_close(clocks[0], torch.zeros(32))
        torch.testing.assert_close(clocks[-1], torch.ones(32), atol=1e-6, rtol=0.0)
        self.assertTrue(bool(((clocks[1:] - clocks[:-1]) >= -1e-7).all()))
        self.assertLessEqual(0.2, 1.0 / math.pi)

    def test_training_clock_also_has_synchronous_endpoints(self):
        base = torch.tensor([0.0, 1.0])
        tau, use_wave, delta, _ = sample_wff_timesteps(
            base, 8, probability=1.0, delta_min=0.2, delta_max=0.2
        )
        torch.testing.assert_close(tau[0], torch.zeros(8), atol=1e-7, rtol=0.0)
        torch.testing.assert_close(tau[1], torch.ones(8), atol=1e-7, rtol=0.0)
        torch.testing.assert_close(use_wave, torch.ones_like(use_wave))
        torch.testing.assert_close(delta, torch.full_like(delta, 0.2))

    def test_zero_delta_wff_step_matches_ordinary_ode(self):
        model = _small_model()
        z = torch.randn(2, 6, 8)
        x_prev = torch.zeros_like(z)
        kwargs = dict(
            model=model,
            z=z,
            x_pred_prev=x_prev,
            config=_Config(),
            cfg_scale=1.0,
            self_cond_cfg_scale=1.0,
            cond_seq=None,
            cond_seq_mask=None,
        )
        z_scalar, x_scalar = _ode_step(t=0.25, t_next=0.30, **kwargs)
        tau = torch.full((6,), 0.25)
        tau_next = torch.full((6,), 0.30)
        z_wff, x_wff = _wff_ode_step(t=tau, t_next=tau_next, **kwargs)
        torch.testing.assert_close(z_wff, z_scalar, rtol=0.0, atol=1e-6)
        torch.testing.assert_close(x_wff, x_scalar, rtol=0.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
