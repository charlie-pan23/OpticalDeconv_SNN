# Frozen Evaluation Snapshot

This directory stores the frozen evaluation state for the HIPSA / OpticalDeconv_SNN experiments.

## Frozen checkpoints

- CIFAR10-DVS: cifar10dvs_best_clip3_b96_wd001_do03_val733_test764.pth
- DVS Gesture: best_dvsgesture_acc88p54.pth

## Clean accuracy

- CIFAR10-DVS eval00: approximately 76.30%
- DVS Gesture eval00: approximately 88.54%

## Activity trace

The activity trace uses schema v2_lif_adc and separates:
- mvm_input_activity
- mvm_output_nonzero_activity
- lif_spike_activity
- adc_request_activity

## Notes

The threshold=0.02 eval02 result is treated as a conservative ADC baseline.
ADC threshold sweep should be reported separately.
