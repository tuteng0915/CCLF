# 3 Method: Progressive Decode Correction

## 3.1 Motivation

ELF exposes a mismatch between early token commitment and late correction. In the anchor probes, the trajectory begins committing around the cliff at \(t \approx 0.25\): at \(t=0.25\), probe v4 reports \(23.6\%\) committed-correct positions, \(41.4\%\) committed-wrong positions, and \(35.0\%\) uncommitted positions (`results/elf/probe_v4/anchor_probe_v4.json`, key `1.0`, fields `committed_correct_mean`, `committed_wrong_mean`, `uncommitted_mean`). Probe v1 shows the same transition in recovery, with top-5 recovery rising from \(12.3\%\) at \(t=0.20\) to \(36.8\%\) at \(t=0.25\) and \(61.1\%\) at \(t=0.30\) (`results/elf/probe_v1/anchor_probe.json`, fields `topk_rec_mean`, `t`). However, the final decode branch retains substantial corrective capacity across the plateau: the decode advantage \(G_{\mathrm{dec}}(t)\) is approximately \(+0.17\) to \(+0.28\) for \(t \in [0.30, 0.95]\) (`results/elf/probe_decode_v1/probe_decode_branch.json`, fields `gap_top1_mean`, `t`). In the final-only ELF architecture, this correction is used only at the endpoint, leaving a trajectory-level signal unused during the interval where many wrong commitments are still recoverable.

## 3.2 Preliminary: Decode Branch and the Two-Pass Readout

Let \(x\) denote the clean T5 contextual embedding sequence and \(\varepsilon\) a standard Gaussian noise sample. ELF uses the linear flow interpolation

\[
z_t = t \cdot x + (1-t) \cdot \varepsilon,
\]

where \(t=0\) is pure noise and \(t=1\) is clean. Given \(z_t\), the denoising network produces a continuous output \(\hat{x}^{\mathrm{den}}_t\) in T5 contextual space through its denoising readout. ELF then applies a decode branch at the endpoint. To probe whether this branch is useful before the endpoint, we define a two-pass readout: feed \(\hat{x}^{\mathrm{den}}_t\) back through the backbone as if it were the input at \(t=1\), and denote the resulting contextual state by \(h^{\mathrm{dec}}_t\). The corresponding decode correction residual is

\[
c_t = h^{\mathrm{dec}}_t - \hat{x}^{\mathrm{den}}_t.
\]

Two empirical facts motivate treating \(c_t\) as a training signal. First, the top-1 decode gap

\[
G_{\mathrm{dec}}(t) =
\mathrm{top1}(p^{\mathrm{dec}}_t) -
\mathrm{top1}(p^{\mathrm{lin}}_t)
\]

is positive throughout the plateau. For example, \(G_{\mathrm{dec}}\) is \(+0.245\) at \(t=0.30\), \(+0.241\) at \(t=0.40\), \(+0.196\) at \(t=0.50\), \(+0.174\) at \(t=0.75\), and \(+0.159\) at \(t=0.95\) (`results/elf/probe_decode_v1/probe_decode_branch.json`, fields `dec_top1_gt_mean`, `lin_top1_gt_mean`, `t`). Second, the interpolation probe at \(t=0.95\) shows monotonic improvement as more of the decode residual is applied: top-1 ground-truth recovery rises from \(0.810\) at interpolation weight \(\gamma=0.0\), to \(0.886\), \(0.940\), \(0.961\), and \(0.969\) at \(\gamma \in \{0.25,0.5,0.75,1.0\}\) (`results/elf/probe_decode_v1/probe_decode_branch.json`, fields `interp_0.0_top1_gt_mean` through `interp_1.0_top1_gt_mean`, index where `t=0.95`). Thus \(c_t\) is not merely an endpoint artifact; it is an empirically valid correction direction through the late trajectory.

## 3.3 Progressive Decode Correction Loss

Progressive Decode Correction (PDC) injects this decode-branch target into training before the endpoint. The auxiliary loss is

\[
\mathcal{L}_{\mathrm{PDC}}(t)
=
\left\|
\hat{x}^{\mathrm{den}}_t -
\mathrm{sg}(h^{\mathrm{dec}}_t)
\right\|^2 \cdot \mu(t),
\]

where \(\mathrm{sg}(\cdot)\) denotes stop-gradient. The decode pass is therefore treated as a fixed target for the denoising output rather than as a second branch to be optimized through. The full objective is

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{ELF}}
+ \lambda \mathcal{L}_{\mathrm{PDC}}.
\]

The schedule \(\mu(t)\) is derived from the measured decode advantage \(G_{\mathrm{dec}}(t)\). We set \(\mu(t)=0\) for \(t<0.25\), since the pre-cliff signal is dominated by noisy wrong commitments; \(\mu(t)\propto G_{\mathrm{dec}}(t)\) for \(t \in [0.25,0.95)\); and \(\mu(t)=0\) for \(t\ge 0.95\), avoiding the known two-pass artifact at \(t=1.0\). The JSON-derived fit is

\[
\mu(t) \approx
0.8340 \cdot (t-0.25)^{0.0197}
\cdot (0.95-t)^{0.1027},
\]

for \(t \in [0.25,0.95)\), and zero outside this window (`results/elf/pdc_schedule/mu_t_schedule.json`, fields `fit`, `t`, `mu_t`, and `gdec_source`). The normalized schedule peaks at \(t=0.35\) with \(\mu(t)=1.0\), matching the maximum measured decode gap \(G_{\mathrm{dec}}=0.276\) at the same timestep. Intuitively, this loss guides the denoising trajectory to internalize the decode correction residual, so that by the time \(t\) reaches the plateau, the latent state is already closer to the token manifold without requiring a second pass.

## 3.4 Connection to Fate Tracking

Probe v4 directly tracks the fate of positions that are committed to the wrong token at intermediate source times. The relevant quantities are stored in `results/elf/probe_v4/anchor_probe_v4.json`, key `1.0`, subkey `fate`.

| source_t | n_wrong | traj_corrected | decode_corrected | stays_wrong |
|---:|---:|---:|---:|---:|
| 0.25 | 69.7 | 73.9% | 22.2% | 3.9% |
| 0.40 | 39.6 | 38.5% | 56.8% | 4.7% |
| 0.50 | 30.6 | 24.0% | 71.8% | 4.3% |

The table shows a shift in where correction occurs. At \(t=0.25\), most wrong commitments that are later repaired are corrected by the remaining denoising trajectory. By \(t=0.50\), only \(24.0\%\) are corrected by the trajectory itself, while \(71.8\%\) are corrected only by the decode branch. The residual error that stays wrong remains near \(4\%\). PDC targets exactly this regime: it does not claim to reduce the irreducible remainder, but it attempts to move decode-only corrections into the denoising path before the endpoint. This is a method hypothesis to be tested by training, not yet a demonstrated perplexity or generation-quality improvement.

## 3.5 Training Details

Training details TBD. This section will be filled after implementation of the PDC fine-tuning objective and ablations over \(\lambda\), the \(\mu(t)\) schedule, and whether \(h^{\mathrm{dec}}_t\) is recomputed online or cached (see spec-04).
