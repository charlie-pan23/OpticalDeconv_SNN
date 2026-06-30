class ADCProxyCounter:

    def __init__(self):
        self.total_requests = 0
        self.layer_requests = {}

    def update_from_spikes(self, spike_dict):

        for layer, spikes in spike_dict.items():

            # proxy: spike → ADC request
            req = (spikes > 0).sum().item()

            self.total_requests += req

            if layer not in self.layer_requests:
                self.layer_requests[layer] = 0

            self.layer_requests[layer] += req

    def finalize(self):

        return {
            "total_requests": self.total_requests,
            "layer_requests": self.layer_requests
        }