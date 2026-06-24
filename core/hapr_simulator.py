import torch
from utils.Logger import logger


class HAPRSimulator:
    """
    Hierarchical Analog-Photonic Reduction (HAPR) 仿真器
    功能：模拟光电探测器(PD)输出端的光电流汇聚，以及阈值比较器截断。
    从而推算共享 ADC 池的真实激活占空比。
    """

    def __init__(self, model, threshold=0.05):
        """
        :param threshold: 模拟比较器的噪声/激活阈值。
                          光电流绝对值低于此阈值的通道将被静默，不触发 ADC。
        """
        self.model = model
        self.threshold = threshold
        self.total_adc_triggers = 0
        self.total_analog_channels = 0
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """
        HAPR 发生在光子 MVM 之后、LIF 之前。
        因此，我们将钩子挂载到映射为光子阵列的核心卷积层上。
        """
        for name, module in self.model.named_modules():
            # 这里的命名依据 models/snn_vgg.py 和 snn_cnn.py 的物理映射设定
            if 'conv3' in name or 'conv4' in name or 'deconv_core' in name:
                hook = module.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)
                logger.debug(f"[HAPR Simulator] Analog threshold hook registered on: {name}")

    def _hook_fn(self, module, input, output):
        """
        拦截卷积层的输出（在物理上对应光电转换后的模拟电压/电流）
        """
        # 只有绝对值大于模拟阈值的通道，才会向后级的 Analog MUX 发出 ADC 转换请求
        active_channels = torch.sum(torch.abs(output) > self.threshold).item()

        self.total_adc_triggers += active_channels
        self.total_analog_channels += output.numel()

    def get_adc_activity(self):
        """获取真实触发 ADC 的占空比 (α_ADC)"""
        if self.total_analog_channels == 0:
            return 0.0
        return self.total_adc_triggers / self.total_analog_channels

    def clear(self):
        self.total_adc_triggers = 0
        self.total_analog_channels = 0

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()