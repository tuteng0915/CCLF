"""Smoke test for spec-11 Diffusion Forcing implementation."""
import sys; sys.path.insert(0, 'src')
import torch, math

from modules.model import ELF_B
# Baseline checkpoint: text_encoder_dim=512, bottleneck_dim=128, vocab_size=0 (no lin_branch)
# proj_kernel: (768, 512), unembed_kernel: (512, 32100) from text_encoder_dim=512
model = ELF_B(text_encoder_dim=512, max_length=128, bottleneck_dim=128,
              vocab_size=32100, num_model_mode_tokens=4)

# Load with strict=False to skip lin_branch mismatch (baseline has no lin_branch in checkpoint)
ckpt = torch.load('converted/elf_b-owt-baseline_torch.pt', map_location='cpu')
missing, unexpected = model.load_state_dict(ckpt['params'], strict=False)
print(f"Missing: {missing[:3]}, Unexpected: {unexpected[:3]}")

model.eval().cuda()
print("Model loaded OK")

B, L, d = 4, 128, 512
cond = torch.zeros(B, L, d).cuda()
mask = torch.zeros(B, L).cuda()

class MockConfig:
    use_bf16 = True

config = MockConfig()

# Run a real ODE step to get meaningful x_pred
# Must pass self_cond_cfg_scale so SC-CFG tokens are prepended (RoPE expects full 140 tokens)
z = torch.randn(B, L, 1024).cuda()
t_batch = torch.full((B,), 0.5).cuda()
sc_scale = torch.ones(B, dtype=torch.float32).cuda()
with torch.no_grad():
    x_pred_real, _, _ = model(z, t_batch, decoder_step_active=None, self_cond_cfg_scale=sc_scale)
print(f"x_pred shape: {x_pred_real.shape}")

# Test _get_df_entropy
from utils.generation_utils import _get_df_entropy, _apply_df_step
H = _get_df_entropy(x_pred_real, model, config, cond, mask)
print(f"Entropy shape: {H.shape}")
print(f"Entropy at t=0.5: mean={H.mean():.3f} min={H.min():.3f} max={H.max():.3f}")
print(f"Max possible H = log(32100) = {math.log(32100):.3f}")
print(f"H<0.5: {(H<0.5).float().mean()*100:.1f}%  H<1.0: {(H<1.0).float().mean()*100:.1f}%  H<2.0: {(H<2.0).float().mean()*100:.1f}%")

# Also test at t=0.9
t_09 = torch.full((B,), 0.9).cuda()
with torch.no_grad():
    x_pred_09, _, _ = model(z, t_09, decoder_step_active=None, self_cond_cfg_scale=sc_scale)
H09 = _get_df_entropy(x_pred_09, model, config, cond, mask)
print(f"Entropy at t=0.9: mean={H09.mean():.3f} | H<1.0: {(H09<1.0).float().mean()*100:.1f}%")

# Test _apply_df_step freeze
from configs.config import SamplingConfig
sc_freeze = SamplingConfig()
sc_freeze.df_variant = 'freeze'
sc_freeze.df_commit_thresh = 1.0

z_next = torch.randn(B, L, d).cuda()
z_new = _apply_df_step(z_next, x_pred_real, 0.5, model, config, cond, mask, sc_freeze)
frozen_frac = (z_new - x_pred_real).norm(dim=-1) < 1e-4
print(f"Frozen positions (thresh=1.0, t=0.5): {frozen_frac.float().mean()*100:.1f}%")

# Test _apply_df_step soft
sc_soft = SamplingConfig()
sc_soft.df_variant = 'soft'
sc_soft.df_soft_alpha = 0.5
z_new2 = _apply_df_step(z_next, x_pred_real, 0.5, model, config, cond, mask, sc_soft)
diff = (z_new2 - z_next).abs().mean()
print(f"Soft: mean |z_new - z_next| = {diff:.5f} (should be > 0)")

print("\nSmoke test PASSED")
