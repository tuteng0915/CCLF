import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_step import _kd_kl_per_token, _kd_omega_gate


class KDTest(unittest.TestCase):
    def test_kd_kl_is_zero_for_identical_logits(self):
        logits = torch.tensor([[[1.0, -0.5, 0.25]]])
        value = _kd_kl_per_token(logits, logits, temperature=4.0)
        torch.testing.assert_close(value, torch.zeros_like(value), atol=1e-6, rtol=0.0)

    def test_kd_kl_matches_manual_teacher_student_direction(self):
        teacher = torch.tensor([[[3.0, 0.0, -1.0]]])
        student = torch.tensor([[[-1.0, 0.0, 3.0]]], requires_grad=True)
        value = _kd_kl_per_token(teacher, student, temperature=2.0)
        self.assertGreater(value.item(), 0.0)
        value.sum().backward()
        self.assertIsNotNone(student.grad)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_kd_gate_is_plateau_shaped_and_symmetric(self):
        t = torch.tensor([0.0, 0.25, 0.60, 0.95, 1.0])
        gate = _kd_omega_gate(t, k=10.0, t_low=0.25, t_high=0.95)
        self.assertGreater(gate[2], gate[1])
        self.assertGreater(gate[2], gate[3])
        torch.testing.assert_close(gate[1], gate[3], atol=1e-6, rtol=0.0)
        self.assertLess(gate[0], gate[1])
        self.assertLess(gate[4], gate[3])


if __name__ == "__main__":
    unittest.main()
