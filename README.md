### Stage 0: clean accuracy sanity check

```bash
python eval/eval_00_clean_accuracy.py \
  --config ./configs/config_cifar10dvs_clip3_b96_wd001_do03.yaml \
  --checkpoint ./results/cifar10dvs/cifar10dvs_best_clip3_b96_wd001_do03_val733_test764.pth \
  --batch-size 256
```
```bash
python eval/eval_00_clean_accuracy.py \
  --config ./results/dvsgesture/config_dvsgesture_acc88p54.yaml \
  --checkpoint ./results/dvsgesture/best_dvsgesture_acc88p54.pth \
  --batch-size 128
```

CIFAR10-DVS: PASS, 76.30%\
DVS Gesture: PASS, 88.54%\

**PASS, CIFAR 的 accuracy 对上了，但 loss 没完全对上。**

### Stage 1: activity trace\

```bash
python eval/eval_01_activity_trace.py \
  --config ./configs/config_cifar10dvs_clip3_b96_wd001_do03.yaml \
  --checkpoint ./results/cifar10dvs/cifar10dvs_best_clip3_b96_wd001_do03_val733_test764.pth \
  --batch-size 128
```
```bash
python eval/eval_01_activity_trace.py \
  --config ./results/dvsgesture/config_dvsgesture_acc88p54.yaml \
  --checkpoint ./results/dvsgesture/best_dvsgesture_acc88p54.pth \
  --batch-size 128
```
```
CIFAR10-DVS / clipped_count max=3 / SpikingVGGGAP\
Dense SOP/image  = 5.38 G\
Active SOP/image = 0.526 G\
Active SOP ratio = 9.77%
```
```
DVS Gesture / binary / SpikingGestureCNN\
Dense SOP/image  = 2.74 G\
Active SOP/image = 0.236 G\
Active SOP ratio = 8.63%
```
**PASS,\
active SOP 和 LIF spike activity 已经可信；\
mvm_output_nonzero_activity 只是 debug 字段；\
adc_request_activity 在当前 threshold=0.02 下偏高，需要 threshold sensitivity。**

### Stage 2: power/performance

```bash
python eval/eval_02_power_perf.py \
  --config ./configs/config_cifar10dvs_clip3_b96_wd001_do03.yaml \
  --checkpoint ./results/cifar10dvs/cifar10dvs_best_clip3_b96_wd001_do03_val733_test764.pth \
  --activity results/cifar10dvs/eval/eval01_activity_trace/eval01_activity_trace.json \
  --hardware configs/hardware_hipsa.yaml \
  --device-params configs/device_params.yaml
```
```bash
python eval/eval_02_power_perf.py \
  --config ./results/dvsgesture/config_dvsgesture_acc88p54.yaml \
  --checkpoint ./results/dvsgesture/best_dvsgesture_acc88p54.pth \
  --activity results/dvsgesture/eval/eval01_activity_trace/eval01_activity_trace.json \
  --hardware configs/hardware_hipsa.yaml \
  --device-params configs/device_params.yaml
```
```
CIFAR10-DVS 

Latency / image    = 80.23 us
Throughput         = 12,464 images/s
Total power        = 2.34 W
Energy / image     = 187.70 uJ

ADC macro utilization = 95.57%
ADC saturated         = false
ADC stall cycles      = 0
```
```
DVS Gesture

Latency / image    = 36.06 us
Throughput         = 27,732 images/s
Total power        = 2.27 W
Energy / image     = 81.69 uJ

ADC macro utilization = 73.28%
ADC saturated         = false
ADC stall cycles      = 0
```
**PASS, conservative ADC baseline.**


### Stage 3: robustness
等待 eval02 和 eval01 稳定