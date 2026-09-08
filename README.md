# OpticalDeconv_SNN / HIPSA Evaluation README

This branch (`5090`) contains the post-training evaluation pipeline for the HIPSA paper's device-calibrated evaluation section. The evaluation is organized as a traceable chain:

```text
frozen checkpoint
  -> clean accuracy
  -> activity trace
  -> active-SOP modeling
  -> latency / power / energy modeling
  -> ADC/HAPR sensitivity
  -> device-specific robustness
  -> paper figures
```

## Repository layout

```text
configs/             Workload, HIPSA hardware, and device-parameter configs
eval/                Evaluation data-generation scripts, eval_00 to eval_06
plot/                Plotting scripts; plots read saved JSON/CSV only
results/             Checkpoints, train/test artifacts, and eval_v2 outputs
hardware/            Older hardware helpers; some are superseded by eval_v2 scripts
utils/               Shared checkpoint/config/data/result-I/O helpers
scripts/             Utility scripts
frozen_artifacts/    Frozen post-training artifacts
```

Evaluation data is written to:

```text
results/eval_v2/<dataset>/eval_xx/
```

Figures are written to:

```text
plot/results/<stage_or_final>/
```

Final paper-style figures are written to:

```text
plot/results/final/
```

## Frozen workloads

| Dataset | Model | Input encoding | T | Test samples | Role | Checkpoint |
|---|---|---:|---:|---:|---|---|
| CIFAR10-DVS | `SpikingVGGGAP` | `clipped_count max=3` | 10 | 1000 | Count-coded high-activity stress workload | `results/cifar10dvs/cifar10dvs_best_clip3_b96_wd001_do03_val733_test764.pth` |
| DVS Gesture | `SpikingGestureCNN` | binary | 10 | 288 | Strict binary event-stream workload | `results/dvsgesture/best_dvsgesture_acc88p54.pth` |

Important interpretation note:

- CIFAR10-DVS is **not** a strict 1-bit binary spike workload in the current main checkpoint. It should be described as a count-coded / high-activity stress workload.
- DVS Gesture is the strict binary event-stream workload.
- If the paper needs a strict 1-bit CIFAR10-DVS claim, a separate binary CIFAR10-DVS baseline should be trained and evaluated.

## Hardware-modeling scope

All HIPSA numbers in this branch are **device-calibrated architecture-level estimates**, not fabricated-silicon measurements.

Main modeling assumptions:

- Four logical `64 x 64` PDPU tiles.
- Effective clock / utilization: `1 GHz / 40%`.
- Realized active rate: `6.5536 TSOP/s`.
- CW laser is on during inference.
- MRR weights are calibrated / static during inference.
- No timestep-level optical power-gating is hidden in the main result.
- Continuous per-ring MRR thermal locking is excluded from the main inference-time result and treated as a stress case.

The corrected publication candidate is:

```text
HAPR group size = 4
ADC macros      = 8
```

The superseded exploratory point is:

```text
HAPR group size = 8
ADC macros      = 16
```

G4/A8 is the corrected paper-facing point: four physical 64x64 tiles produce 64 post-HAPR lanes, while eight 10 GS/s ADC macros provide an architecture-level 80-sample/ns admission capacity. This is not circuit timing closure.

## Evaluation stages

### eval_00: clean accuracy sanity check

Script: `eval/eval_00.py`

Purpose:

- Load frozen checkpoint and config.
- Run the frozen test split.
- Save clean accuracy, loss, predictions, per-class accuracy, and confusion matrix.

Main outputs:

```text
results/eval_v2/<dataset>/eval_00/summary.json
results/eval_v2/<dataset>/eval_00/predictions.csv
results/eval_v2/<dataset>/eval_00/confusion_matrix.csv
results/eval_v2/<dataset>/eval_00/per_class_accuracy.csv
```

Current results:

| Dataset | Accuracy | Loss |
|---|---:|---:|
| CIFAR10-DVS | 76.30% | 0.748987 |
| DVS Gesture | 88.54% | 0.411062 |

Limitations:

- This stage only validates checkpoint / config / split consistency.
- It does not collect hardware activity.
- CIFAR10-DVS accuracy corresponds to `clipped_count max=3`.

### eval_01: activity trace and active-SOP statistics

Script: `eval/eval_01.py`

Purpose:

- Trace model activity through forward hooks.
- Compute dense SOPs and active SOPs.
- Separate MVM activity, LIF spike activity, and ADC request proxy.

Main outputs:

```text
results/eval_v2/<dataset>/eval_01/summary.json
results/eval_v2/<dataset>/eval_01/sop_summary.json
results/eval_v2/<dataset>/eval_01/layer_activity.csv
results/eval_v2/<dataset>/eval_01/timestep_activity.csv
```

Headline activity:

| Dataset | Dense SOP/image | Active SOP/image | Active SOP ratio | MVM input activity | LIF spike activity | ADC request activity |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR10-DVS | 5.380G | 0.526G | 9.77% | 12.65% | 6.37% | 76.91% |
| DVS Gesture | 2.737G | 0.236G | 8.63% | 10.47% | 4.21% | 65.30% |

Definitions:

- `mvm_input_activity`: primary signal for active-SOP counting.
- `mvm_output_nonzero_activity`: debug field only.
- `lif_spike_activity`: post-LIF spike activity, used as digital/NoC proxy.
- `adc_request_activity`: comparator-request proxy before HAPR/ADC modeling.

Limitations:

- Activity is traced from PyTorch inference, not measured on silicon.
- ADC request is a modeling proxy.
- CIFAR10-DVS activity is based on count-coded input.

### eval_02: default device-calibrated latency / power / energy

Script: `eval/eval_02.py`

Purpose:

- Read eval_01 activity summaries.
- Apply the HIPSA timing and device-calibrated power model.
- Compute latency, throughput, power, energy, and ADC pool behavior.
- Default point: `HAPR=8`, `ADC=16`.

Main outputs:

```text
results/eval_v2/<dataset>/eval_02/summary.json
results/eval_v2/<dataset>/eval_02/latency_energy_summary.json
results/eval_v2/<dataset>/eval_02/power_breakdown.csv
results/eval_v2/<dataset>/eval_02/adc_pool_summary.json
```

Default-point headline results:

| Dataset | Latency | Energy | Power | Throughput | ADC status |
|---|---:|---:|---:|---:|---|
| CIFAR10-DVS | 92.16 us | 239.29 uJ | 2.60 W | 10,851 img/s | saturated |
| DVS Gesture | 79.27 us | 204.17 uJ | 2.58 W | 12,614 img/s | saturated |

Limitations:

- Default `HAPR=8 / ADC=16` is intentionally conservative and ADC-saturated.
- This is not the final balanced paper design point.
- Power is derived from component models and activity factors.
- Energy is model-derived as `power x latency`.

### eval_03: comparator threshold sensitivity

Script: `eval/eval_03.py`

Purpose:

- Sweep comparator threshold.
- Evaluate whether thresholding can reduce ADC pressure.
- Default mode applies threshold-induced zeroing before LIF update.

Main outputs:

```text
results/eval_v2/<dataset>/eval_03/summary.json
results/eval_v2/<dataset>/eval_03/threshold_sweep.csv
results/eval_v2/<dataset>/eval_03/layer_threshold_sweep.csv
```

Main conclusion:

- Naive comparator thresholding is **not** a reliable primary optimization knob.
- Moderate thresholds do not sufficiently relieve ADC saturation.
- Aggressive thresholds reduce activity but quickly damage accuracy.

Limitations:

- This stage is a sensitivity / stress test, not the final optimization method.
- Threshold-induced zeroing is an aggressive proxy.

### eval_04: HAPR / ADC pool / MRR stabilization sensitivity

Script: `eval/eval_04.py`

Purpose:

- Read eval_01 activity summaries.
- Sweep HAPR group size and ADC pool size.
- Evaluate MRR stabilization overhead.
- Identify default, conservative, balanced, and aggressive design points.

Main outputs:

```text
results/eval_v2/<dataset>/eval_04/summary.json
results/eval_v2/<dataset>/eval_04/adc_pool_sweep.csv
results/eval_v2/<dataset>/eval_04/hapr_adc_sweep.csv
results/eval_v2/<dataset>/eval_04/mrr_sensitivity.csv
results/eval_v2/<dataset>/eval_04/selected_design_points.csv
```

Representative design points:

| Design | HAPR | ADC macros | Interpretation |
|---|---:|---:|---|
| Default | 8 | 16 | Conservative but ADC-saturated |
| Conservative | 8 | 64 | Lower HAPR risk, more ADCs |
| Corrected candidate | 4 | 8 | Architecture-level G4/A8 admission point |
| Rejected without temporal storage | 16 | 16 | Not used for the corrected design |

Corrected publication candidate:

| Dataset | Latency | Energy | Power |
|---|---:|---:|---:|
| CIFAR10-DVS | 80.23 us | 179.38 uJ | 2.24 W |
| DVS Gesture | 36.06 us | 86.69 uJ | 2.40 W |

Limitations:

- Activity-scaled ADC power can under-penalize very large ADC pools.
- HAPR=32 is aggressive and should not be used as the default without robustness support.
- eval_04 does not directly simulate HAPR summing noise or TIA dynamic-range limits.
- MRR full thermal-locking cases are stress cases, not main assumptions.

### eval_05: device-specific robustness

Script: `eval/eval_05.py`

Purpose:

- Run hardware-aware forward-only robustness sweeps.
- Inject device-specific perturbations around the photonic/electronic front end.
- Evaluate accuracy drop and energy/activity impact under the balanced design context.

Perturbation types:

- ADC precision: `4 / 5 / 6 / 8 bits`
- MRR transmission perturbation: `1 / 2 / 3 / 5%`
- Laser intensity fluctuation: `1 / 2 / 3%`
- WDM crosstalk: `-30 / -25 / -20 / -15 dB`
- TIA/HAPR output noise: `0.5 / 1 / 2 / 3%`
- Combined stress case

Main outputs:

```text
results/eval_v2/<dataset>/eval_05/summary.json
results/eval_v2/<dataset>/eval_05/robustness_summary.csv
results/eval_v2/<dataset>/eval_05/robustness_detail.csv
```

Paper-facing summary:

- Full sweep results are stored in eval_05 outputs.
- Main paper figure uses representative practical points:
  - ADC 6-bit
  - MRR 3%
  - Laser 3%
  - WDM -20 dB
  - TIA/HAPR 2%
  - Combined stress

Limitations:

- eval_05 is a hardware-aware simulation/proxy, not a fabricated-device measurement.
- Negative energy change can occur when perturbation suppresses activity; this is activity-dependent model behavior, not a physical energy-saving mechanism.
- Full perturbation sweeps should remain in the repository / appendix; main text should use representative points.

### eval_06: CPU/GPU software runtime baseline

Script: `eval/eval_06.py`

Purpose:

- Measure PyTorch CPU and CUDA runtime baselines.
- Use batch size 1 for latency-oriented edge inference.
- Estimate software energy when active platform power is provided.

Main outputs:

```text
results/eval_v2/<dataset>/eval_06/         # runtime outputs
plot/results/eval_06_local/                # local paper-facing summary and figures
plot/results/eval_06_local/plot_06_comparison_data.csv
```

Paper-facing local summary:

| Dataset | Platform | Latency | Energy |
|---|---|---:|---:|
| CIFAR10-DVS | CPU | 57.95 ms | 4262.18 mJ |
| CIFAR10-DVS | GPU | 29.57 ms | 2069.27 mJ |
| CIFAR10-DVS | HIPSA HAPR16/ADC32 | 0.080 ms | 0.179 mJ |
| DVS Gesture | CPU | 41.96 ms | 3028.39 mJ |
| DVS Gesture | GPU | 25.63 ms | 1485.22 mJ |
| DVS Gesture | HIPSA HAPR16/ADC32 | 0.036 ms | 0.0867 mJ |

Limitations:

- CPU/GPU runtime and energy are platform-specific.
- Do not mix AutoDL server results with local laptop CPU/GPU claims.
- Software energy is only meaningful when active power is measured and provided.

## Plot scripts

### Stage plots

| Script | Purpose | Main inputs | Main outputs |
|---|---|---|---|
| `plot/plot_00.py` | Confusion matrix from eval_00 | `results/eval_v2/<dataset>/eval_00/confusion_matrix.csv` | `plot/results/eval_00/` |
| `plot/plot_01.py` | Workload activity | eval_01 summaries | `plot/results/eval_01/` |
| `plot/plot_02.py` | HIPSA latency / energy overview | eval_02 summaries | `plot/results/eval_02/` |
| `plot/plot_03.py` | HIPSA power breakdown | eval_02 power CSVs | `plot/results/eval_02/` |
| `plot/plot_04.py` | Comparator threshold sweep | eval_03 threshold CSVs | `plot/results/eval_03/` |
| `plot/plot_05.py` | HAPR/ADC/MRR sensitivity | eval_04 CSVs | `plot/results/eval_04/` |
| `plot/plot_06.py` | CPU/GPU/HIPSA comparison | eval_06 + eval_04 | `plot/results/eval_06/` |
| `plot/plot_07.py` / `plot_07_*` | Robustness sweeps and summaries | eval_05 robustness CSVs | `plot/results/eval_05*/` |

### Final paper figures

| Script | Figure | Outputs |
|---|---|---|
| `plot/plot_final_fig5.py` | Activity + latency + energy | `fig5_activity_performance.pdf`, `fig5a_activity.pdf`, `fig5b_latency.pdf`, `fig5c_energy.pdf` |
| `plot/plot_final_fig6.py` | Power breakdown + HAPR/ADC design | `fig6_power_adc_hapr.pdf`, `fig6a_power_breakdown.pdf`, `fig6b_hapr_adc_energy.pdf`, `fig6c_adc_utilization.pdf` |
| `plot/plot_final_fig7.py` | Representative robustness summary | `fig7_robustness_summary.pdf`, `fig7_robustness_summary_panel.pdf` |

The combined figures are useful for previewing. The individual panel PDFs are intended for LaTeX `subfigure` / `subcaption` layouts, where subfigure titles and captions can be controlled by the paper template.

## Recommended paper Section 4 mapping

```text
4 Device-Calibrated Evaluation
  4.1 Evaluation Protocol and Frozen Workloads
      Table: workload, encoding, T, samples, checkpoint, accuracy

  4.2 Activity Trace and Active-SOP Modeling
      Fig. 5(a): workload activity

  4.3 Latency, Throughput, and Energy
      Fig. 5(b)(c): CPU/GPU/HIPSA latency and energy

  4.4 Power Breakdown and ADC/HAPR Sensitivity
      Fig. 6(a): balanced power breakdown
      Fig. 6(b)(c): HAPR/ADC energy and ADC backend pressure

  4.5 Device-Specific Robustness
      Fig. 7: representative robustness summary
```

A separate `4.6 Discussion and Limitations` section is optional. With a short paper page limit, limitations are better integrated into the corresponding subsections and summarized briefly in the conclusion.

## Known limitations and next steps

1. **CIFAR10-DVS binary baseline**  
   Current CIFAR10-DVS uses `clipped_count max=3`. Add a binary CIFAR10-DVS baseline if strict 1-bit CIFAR claims are required.

2. **Architecture-level estimates**  
   HIPSA latency/power/energy are device-calibrated estimates, not silicon measurements.

3. **ADC model**  
   Activity-scaled ADC power is useful for design exploration but may under-penalize very large ADC pools. Corrected candidate `HAPR=4 / ADC=8` is the paper-facing point; 10 GS/s is a capacity assumption, not circuit timing closure.

4. **MRR locking**  
   Continuous per-ring thermal locking is excluded from the main inference-time power and should only be reported as a stress case.

5. **Robustness simulation**  
   eval_05 perturbations are proxy injections. They support sensitivity analysis but do not replace device-level measurement or detailed circuit simulation.

6. **CPU/GPU baseline**  
   eval_06 results are platform-specific. The paper-facing local results should be traced to the same machine and active-power measurements used in the final figures.
