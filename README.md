Stage 0: clean accuracy sanity check\
CIFAR10-DVS: PASS, 76.30%\
DVS Gesture: PASS, 88.54%\

Stage 1: activity trace\

CIFAR10-DVS / clipped_count max=3 / SpikingVGGGAP\
Dense SOP/image  = 5.38 G\
Active SOP/image = 0.526 G\
Active SOP ratio = 9.77%

DVS Gesture / binary / SpikingGestureCNN\
Dense SOP/image  = 2.74 G\
Active SOP/image = 0.236 G\
Active SOP ratio = 8.63%


Stage 2: power/performance\
等待 eval01 输出

Stage 3: robustness\
等待 eval02 和 eval01 稳定