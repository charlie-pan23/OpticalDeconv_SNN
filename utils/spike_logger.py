import torch

class SpikeTraceLogger:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.reset()

    def reset(self):
        self.trace = {}
        self.layer_activity = {}
        self.timestep_activity = []

    def reset_step(self):
        self.step_spikes = {}

    def hook_fn(self, name):

        def fn(module, input, output):

            spike = output.detach().cpu()

            if name not in self.step_spikes:
                self.step_spikes[name] = []

            self.step_spikes[name].append(spike)

        return fn

    def attach_hooks(self):

        for name, module in self.model.named_modules():

            if "conv" in name or "fc" in name or "linear" in name:
                h = module.register_forward_hook(self.hook_fn(name))
                self.hooks.append(h)

    def step_finalize(self):

        total_active = 0
        total_neurons = 0

        for k, v in self.step_spikes.items():
            spikes = torch.stack(v)

            active = (spikes > 0).sum().item()
            total = spikes.numel()

            total_active += active
            total_neurons += total

            if k not in self.layer_activity:
                self.layer_activity[k] = []

            self.layer_activity[k].append(active / (total + 1e-9))

        self.timestep_activity.append(
            total_active / (total_neurons + 1e-9)
        )

        self.trace[len(self.timestep_activity)] = self.step_spikes

    def get_step_spikes(self):
        return self.step_spikes

    def get_full_trace(self):
        return self.trace

    def get_statistics(self):

        return {
            "global": {
                "mean_activity": float(
                    sum(self.timestep_activity) / len(self.timestep_activity)
                )
            },
            "layer": {
                k: float(sum(v) / len(v)) for k, v in self.layer_activity.items()
            },
            "timestep": self.timestep_activity
        }