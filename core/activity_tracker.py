import torch
from utils.Logger import logger

class ActivityTracker:
    """
    SNN 脉冲稀疏度侦测器 (Spike Activity Tracker)
    基于 PyTorch Forward Hook 机制，非侵入式地采集网络的真实发放率。
    """
    def __init__(self, model):
        self.model = model
        self.total_spikes = 0
        self.total_elements = 0
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """自动遍历模型，将钩子挂载到所有 LIF 神经元上"""
        for name, module in self.model.named_modules():
            # 严格匹配 snntorch 的 Leaky 神经元
            if 'Leaky' in str(type(module)):
                hook = module.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)
                logger.debug(f"[Tracker] Hook registered on LIF neuron: {name}")

    def _hook_fn(self, module, input, output):
        """
        钩子回调函数：拦截神经元输出，统计非零脉冲数
        注意：snntorch 的 Leaky 返回 (spk, mem)，所以 output[0] 是脉冲张量
        """
        spk = output[0]
        # 统计真实产生的脉冲数 (1的个数)
        self.total_spikes += torch.count_nonzero(spk).item()
        # 统计理论上该层的总元素数 (Dense-equivalent 基础)
        self.total_elements += spk.numel()

    def get_active_ratio(self):
        """获取整个推理过程的平均激活率"""
        if self.total_elements == 0:
            return 0.0
        return self.total_spikes / self.total_elements

    def clear(self):
        """清空统计数据，用于下一次推理"""
        self.total_spikes = 0
        self.total_elements = 0

    def remove_hooks(self):
        """释放内存中的钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()