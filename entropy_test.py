import sys; sys.path.insert(0, 'src')
import torch, math
import torch.nn.functional as F

from modules.model import ELF_B
model = ELF_B(text_encoder_dim=512, max_length=128, bottleneck_dim=512, vocab_size=32100,
              num_model_mode_tokens=4)
ckpt = torch.load('converted/elf_b-owt-baseline_torch.pt', map_location='cpu')
model.load_state_dict(ckpt['params'])
model.eval().cuda()

z = torch.randn(4, 128, 1024).cuda()

with torch.no_grad():
    for tval in [0.3, 0.5, 0.8, 1.0]:
        tb = torch.full((4,), tval).cuda()
        x_pred, _, _ = model(z, tb, decoder_step_active=None)
        logits = x_pred.float() @ model.unembed_kernel.float() + model.unembed_bias.float()
        log_p = F.log_softmax(logits, dim=-1)
        H = -(torch.exp(log_p) * log_p).sum(dim=-1)
        print(f't={tval}: H mean={H.mean():.3f} min={H.min():.3f} max={H.max():.3f} | '
              f'H<0.5: {(H<0.5).float().mean()*100:.1f}% | '
              f'H<1.0: {(H<1.0).float().mean()*100:.1f}% | '
              f'H<2.0: {(H<2.0).float().mean()*100:.1f}%')
