# OpticalDeconv_SNN / HIPSA Evaluation Notes (branch `5090`)

This branch contains the post-training evaluation pipeline for the HIPSA paper section on device-calibrated evaluation. The current goal is not to train the strongest SNN model, but to produce a traceable evidence chain from trained checkpoints to activity traces, hardware latency/power estimates, sensitivity sweeps, and paper figures.

## Repository layout

```text
configs/             Hardware, device, and workload configs
eval/                Evaluation data-generation scripts, eval_00 to eval_06
plot/                Plotting scripts; plots read saved JSON/CSV only
results/             Checkpoints, frozen train/test outputs, eval_v2 outputs
hardware/            Older hardware helper modules; some are now superseded
utils/               Shared checkpoint/config/data/result I/O helpers
scripts/             Utility checks and workflow scripts
frozen_artifacts/    Frozen post-training artifacts
```

Evaluation data is written to:

```text
results/eval_v2/<dataset>/eval_xx/
```

Figures are written to:

```text
plot/results/eval_xx/
```

## Frozen workloads

| Dataset | Model | Input encoding | T | Test samples | Current checkpoint |
|---|---|---:|---:|---:|---|
| CIFAR10-DVS | SpikingVGGGAP | clipped_count max=3 | 10 | 1000 | `results/cifar10dvs/cifar10dvs_best_clip3_b96_wd001_do03_val733_test764.pth` |
| DVS Gesture | SpikingGestureCNN | binary | 10 | 288 | `results/dvsgesture/best_dvsgesture_acc88p54.pth` |

Important paper note: CIFAR10-DVS is currently a **count-coded high-activity stress workload**, not a strict 1-bit binary spike workload. DVS Gesture is the strict binary event-stream workload. A CIFAR10-DVS binary baseline should be added later if the paper needs a strict 1-bit CIFAR claim.

## Evaluation pipeline

| Stage | Script | Purpose | Main outputs |
|---|---|---|---|
| eval_00 | `eval/eval_00.py` | Clean accuracy sanity check | `summary.json`, `predictions.csv`, `confusion_matrix.csv` |
| eval_01 | `eval/eval_01.py` | Activity trace and active SOP statistics | `summary.json`, `sop_summary.json`, `layer_activity.csv` |
| eval_02 | `eval/eval_02.py` | Device-calibrated latency / power / energy model | `latency_energy_summary.json`, `power_breakdown.csv` |
| eval_03 | `eval/eval_03.py` | Comparator threshold sensitivity | `threshold_sweep.csv` |
| eval_04 | `eval/eval_04.py` | HAPR / ADC pool / MRR stabilization sensitivity | `hapr_adc_sweep.csv`, `mrr_sensitivity.csv` |
| eval_05 | pending | Device-specific robustness | MRR / laser / WDM / ADC / TIA sensitivity |
| eval_06 | `eval/eval_06.py` | CPU/GPU software runtime baseline | `runtime_summary.json`, `runtime_samples.csv` |

## Reproduce current evaluation

### eval_00: clean accuracy

```bash
python -m eval.eval_00 \
  --config configs/config_cifar10dvs_clip3_b96_wd001_do03.yaml \
  --checkpoint results/cifar10dvs/cifar10dvs_best_clip3_b96_wd001_do03_val733_test764.pth \
  --output-root results/eval_v2 \
  --split test \
  --batch-size 128 \
  --num-workers 4 \
  --device auto \
  --save-logits

python -m eval.eval_00 \
  --config results/dvsgesture/config_dvsgesture_acc88p54.yaml \
  --checkpoint results/dvsgesture/best_dvsgesture_acc88p54.pth \
  --output-root results/eval_v2 \
  --split test \
  --batch-size 128 \
  --num-workers 4 \
  --device auto \
  --save-logits
```

Current results:

| Dataset | Accuracy | Loss |
|---|---:|---:|
| CIFAR10-DVS | 76.30% | 0.748987 |
| DVS Gesture | 88.54% | 0.411062 |

### eval_01: activity trace

```bash
python -m eval.eval_01 \
  --config configs/config_cifar10dvs_clip3_b96_wd001_do03.yaml \
  --checkpoint results/cifar10dvs/cifar10dvs_best_clip3_b96_wd001_do03_val733_test764.pth \
  --output-root results/eval_v2 \
  --split test \
  --batch-size 128 \
  --num-workers 4 \
  --device auto

python -m eval.eval_01 \
  --config results/dvsgesture/config_dvsgesture_acc88p54.yaml \
  --checkpoint results/dvsgesture/best_dvsgesture_acc88p54.pth \
  --output-root results/eval_v2 \
  --split test \
  --batch-size 128 \
  --num-workers 4 \
  --device auto
```

Current activity summary:

| Dataset | Dense SOP/image | Active SOP/image | Active SOP ratio | MVM input activity | LIF spike activity | ADC request activity |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR10-DVS | 5.380G | 0.526G | 9.77% | 12.65% | 6.37% | 76.91% |
| DVS Gesture | 2.737G | 0.236G | 8.63% | 10.47% | 4.21% | 65.30% |

Definitions:

- `mvm_input_activity`: primary signal for active SOP counting.
- `mvm_output_nonzero_activity`: debug field only.
- `lif_spike_activity`: digital spike / NoC proxy.
- `adc_request_activity`: comparator-request proxy before HAPR/ADC modeling.

### eval_02: default device-calibrated model

```bash
python -m eval.eval_02 \
  --datasets cifar10dvs dvsgesture \
  --input-root results/eval_v2 \
  --output-root results/eval_v2 \
  --hardware configs/hardware_hipsa.yaml \
  --device-params configs/device_params.yaml
```

Default point: HAPR group size = 8, ADC macros = 16. This point is conservative but ADC-saturated.

| Dataset | Latency | Energy | Power | Throughput | ADC saturated |
|---|---:|---:|---:|---:|---:|
| CIFAR10-DVS | 92.16 us | 239.29 uJ | 2.60 W | 10,850 img/s | yes |
| DVS Gesture | 79.27 us | 204.17 uJ | 2.58 W | 12,614 img/s | yes |

### eval_03: comparator threshold sensitivity

```bash
python -m eval.eval_03 \
  --dataset cifar10dvs \
  --config configs/config_cifar10dvs_clip3_b96_wd001_do03.yaml \
  --checkpoint results/cifar10dvs/cifar10dvs_best_clip3_b96_wd001_do03_val733_test764.pth \
  --output-root results/eval_v2 \
  --thresholds 0.0 0.01 0.02 0.05 0.10 0.20 \
  --batch-size 128 \
  --num-workers 4 \
  --device auto

python -m eval.eval_03 \
  --dataset dvsgesture \
  --config results/dvsgesture/config_dvsgesture_acc88p54.yaml \
  --checkpoint results/dvsgesture/best_dvsgesture_acc88p54.pth \
  --output-root results/eval_v2 \
  --thresholds 0.0 0.01 0.02 0.05 0.10 0.20 \
  --batch-size 128 \
  --num-workers 4 \
  --device auto
```

Conclusion: aggressive comparator thresholding is not a good primary optimization knob. It quickly damages accuracy while ADC utilization remains high under moderate thresholds. Treat this as sensitivity / stress evidence, not as the main performance-improvement mechanism.

### eval_04: HAPR / ADC / MRR sensitivity

```bash
python -m eval.eval_04 \
  --datasets cifar10dvs dvsgesture \
  --input-root results/eval_v2 \
  --output-root results/eval_v2 \
  --hardware configs/hardware_hipsa.yaml \
  --device-params configs/device_params.yaml \
  --adc-pool-sizes 8 16 32 64 128 \
  --hapr-group-sizes 4 8 16 32
```

Key design insight: the default 16-ADC backend is saturated. HAPR/ADC co-design removes the ADC bottleneck and restores SOP-bound latency.

Recommended provisional main point:

| Design | CIFAR10-DVS | DVS Gesture | Comment |
|---|---:|---:|---|
| HAPR=8, ADC=64 | 80.23 us / 197.09 uJ | 36.06 us / 99.11 uJ | conservative HAPR, more ADCs |
| **HAPR=16, ADC=32** | **80.23 us / 179.38 uJ** | **36.06 us / 86.69 uJ** | **balanced main candidate** |
| HAPR=32, ADC=16 | 80.23 us / 170.52 uJ | 36.06 us / 80.46 uJ | energy-oriented, but aggressive HAPR |

Do not blindly use the auto-selected `HAPR=32, ADC=128` point as the final design. Under the current activity-scaled ADC power model, overprovisioned ADCs are not penalized enough by static bias/clock/area overhead. A later final pass should choose a balanced point manually.

MRR stabilization: practical sub-watt cases should be plotted separately from the full per-ring thermal-locking stress case. The full-locking case is an upper-bound warning, not the main inference-time assumption.

### eval_06: CPU/GPU software baseline

```bash
python -m eval.eval_06 \
  --dataset cifar10dvs \
  --config configs/config_cifar10dvs_clip3_b96_wd001_do03.yaml \
  --checkpoint results/cifar10dvs/cifar10dvs_best_clip3_b96_w001_do03_val733_test764.pth \
  --output-root results/eval_v2 \
  --devices cpu cuda \
  --batch-size 1 \
  --num-workers 0 \
  --num-warmup 20 \
  --num-runs 100
```

Use the actual checkpoint path if the command above is copied manually:

```text
results/cifar10dvs/cifar10dvs_best_clip3_b96_wd001_do03_val733_test764.pth
```

Current AutoDL server results, latency only:

| Dataset | CPU latency | CUDA latency | Note |
|---|---:|---:|---|
| CIFAR10-DVS | 48.60 ms/image | 10.17 ms/image | 100 timed samples |
| DVS Gesture | 38.39 ms/image | 7.73 ms/image | 100 timed samples |

Energy is not reported because active CPU/GPU power was not supplied. Do not use these results as laptop CPU/GPU baselines unless they are rerun on the laptop hardware that will be named in the paper.

## Plotting

Representative plot commands:

```bash
python -m plot.plot_01 --input-root results/eval_v2 --output-root plot/results/eval_01 --datasets cifar10dvs dvsgesture
python -m plot.plot_02 --input-root results/eval_v2 --output-root plot/results/eval_02 --datasets cifar10dvs dvsgesture
python -m plot.plot_03 --input-root results/eval_v2 --output-root plot/results/eval_02 --datasets cifar10dvs dvsgesture
python -m plot.plot_04 --input-root results/eval_v2 --output-root plot/results/eval_03 --datasets cifar10dvs dvsgesture
python -m plot.plot_05 --input-root results/eval_v2 --output-root plot/results/eval_04 --datasets cifar10dvs dvsgesture
python -m plot.plot_06 --input-root results/eval_v2 --output-root plot/results/eval_06 --datasets cifar10dvs dvsgesture --hipsa-hapr 16 --hipsa-adc 32
```

Final paper figures should be redrawn from the saved CSV/JSON outputs. Do not directly use the first generated plots without checking axis scale, units, and selected design points.

## Paper work that can start now

The following parts of Section 4 can already be drafted:

1. Workload and checkpoint protocol: frozen checkpoints, T=10, batch size 1 for latency-oriented evaluation.
2. Clean accuracy table: eval_00 results.
3. Activity trace and active SOP analysis: eval_01 results.
4. Device-calibrated default model: eval_02, with explicit ADC saturation caveat.
5. Comparator threshold sensitivity: eval_03, interpreted as a stress/sensitivity result.
6. HAPR/ADC design-space discussion: eval_04, with HAPR=16 / ADC=32 as provisional balanced point.
7. CPU/GPU baseline methodology: eval_06, but final platform numbers should be rerun on the hardware named in the paper.

## Remaining work before final paper numbers

- Implement eval_05 device-specific robustness: MRR perturbation, laser fluctuation, WDM crosstalk, ADC precision, TIA/HAPR noise.
- Rerun a final eval_02-style summary using the chosen final main design, probably HAPR=16 / ADC=32.
- Add all-biased ADC upper-bound or explicitly state that current ADC power is activity-scaled.
- Add or report a CIFAR10-DVS binary baseline if strict 1-bit CIFAR hardware-facing claims are needed.
- Redraw final figures with paper-ready labels and units.
- Keep full per-ring thermal locking as a stress table, not as a normal curve on the same axis as practical sub-watt overheads.

## Current interpretation

The current evaluation already supports the following narrative:

> HIPSA gains come from exploiting SNN event sparsity and photonic MVM throughput. The default ADC backend is conservative and can become saturated, but HAPR/ADC co-design restores SOP-bound latency. Comparator thresholding alone is not a reliable optimization knob because it harms accuracy early. The final design should use conservative thresholding, balanced HAPR/ADC pooling, and device-specific robustness analysis.
