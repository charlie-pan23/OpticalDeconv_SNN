# HIPSA Frozen Evaluation Package

Tag: hipsa_eval_v2_after_train

## Main frozen checkpoints

### CIFAR10-DVS
- Model: SpikingVGGGAP
- Timestep: T=10
- Encoding: clipped_count max=3
- Best Val: 73.30%
- Test Acc: 76.40%
- Test Loss: 0.9317
- Checkpoint: checkpoints/cifar10dvs/cifar10dvs_best_clip3_b96_wd001_do03_val733_test764.pth
- Config: configs/cifar10dvs/config_cifar10dvs_clip3_b96_wd001_do03.yaml

### DVS Gesture
- Model: SpikingGestureCNN
- Timestep: T=10
- Encoding: binary
- Best Val: 91.67%
- Test Acc: 88.54%
- Test Loss: 0.4111
- Train / Val / Test samples: 1056 / 120 / 288
- Best checkpoint epoch: 46
- Checkpoint: checkpoints/dvsgesture/best_dvsgesture_acc88p54.pth
- Config: configs/dvsgesture/config_dvsgesture_acc88p54.yaml

## Notes

This package freezes training outputs before re-running or rewriting evaluation.
Evaluation scripts inside eval_scripts_snapshot are snapshots only and should be audited before final paper results.
