# HIPSA V3 selective migration and audit

Date: 2026-09-07

## Purpose

This document records the selective migration from
`D:\AgentNotes\HIPSA\HIPSA_revision_package` into the formal repository
`D:\ProgramProject_Hub\OpticalDeconv_SNN`. The revision package was **not**
merged wholesale because it mixed useful evaluation safeguards with changes to
the original project structure, model/data provenance, and paper-facing claims.

## Preserved source of truth

- The formal repository's original `eval/eval_00.py`--`eval/eval_06.py` flow was not replaced.
- `datasets/`, `models/`, `train/`, `results/`, and `plot/` were not overwritten.
- The revision snapshot remains separately archived under `D:\AgentNotes\HIPSA\V3-1`.
- The migrated scripts are additive `eval_06_selected_point.py` and later review/evidence entry points.

## Adopted evaluation changes

1. **Spatial HAPR legality**: G4 is the selected fan-in for four physical tiles; G8/G16 are rejected when temporal analog storage is absent.
2. **Same-neuron partial-sum mapping**: HAPR groups carry explicit neuron/timestep/tile identifiers and are checked for cross-neuron mixing.
3. **ADC admission capacity**: A8 is 8 macros x 10 GS/s = 80 samples/ns against 64 post-HAPR lanes; A6 is explicitly reported as a 4-request/ns shortfall.
4. **Queue/backpressure accounting**: ADC requests are preserved, and finite-FIFO stalls are included in observed cycles and service latency.
5. **State/update accounting**: lazy-LIF, SRAM traffic, fixed-point membrane, and transaction provenance are separate from ADC macro count.
6. **Configuration overhead**: tile-load events, cycles, SRAM traffic, and energy are explicit; the 256-cycle value is labeled an assumption and a sensitivity registry is included.
7. **MRR lock stress cases**: 0%, 1%, 5%, and 10% lock fractions are modeled separately from the nominal headline.
8. **Evidence gates**: review-gate, parameter-audit, split-manifest, RTL-frequency, HAPR Monte Carlo, and design-space entry points are additive and produce provenance-bearing outputs.
9. **Claim boundaries**: 10 GS/s is architecture-level admission capacity, not mux/aperture/TIA circuit timing closure; area is a component-footprint lower bound; digital synthesis is a pre-layout anchor; Transformer is a representative workload.

## Deliberately not adopted as claims

- No claim of full-chip monolithic CMOS--SiPh layout.
- No claim that 4.615 mm² is die/core/layout area.
- No use of CIFAR10-DVS FP32 accuracy as hardware-aware accuracy.
- No claim of broad model-family generality from three workloads on one dataset.
- No headline that hides nonzero MRR lock power.

## Commands used for local validation

```text
python -m compileall hardware eval
python -m unittest discover -s tests -p "test_*.py" -v
python eval/eval_06_selected_point.py --help
python eval/eval_08_validation.py --help
python eval/eval_16_per_sample_adc_queue.py --help
```

The full training/evaluation suite still depends on the repository's optional
ML stack and existing result artifacts. The pure-Python architecture checks in
`tests/test_hipsa_v3_architecture.py` are intended to run without pytest.
