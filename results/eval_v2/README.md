# Evaluation V2 Results

This directory stores regenerated evaluation data for HIPSA Section 4.

## Rule

Evaluation scripts collect and save data only.
Plot scripts read saved data and generate figures under `plot/results/`.

## Mapping

- `eval_00`: clean accuracy sanity check
- `eval_01`: activity trace and active SOP statistics
- `eval_02`: latency, throughput, power, and energy model
- `eval_03`: comparator threshold sweep
- `eval_04`: HAPR, ADC pool, and MRR stabilization sensitivity
- `eval_05`: device-specific robustness
- `eval_06`: CPU/GPU runtime baseline