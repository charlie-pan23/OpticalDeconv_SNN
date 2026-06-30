# hardware aware inference profiling

import os
import json
import torch
import numpy as np

from utils.spike_logger import SpikeTraceLogger
from utils.adc_proxy import ADCProxyCounter


def run_eval(model, dataloader, device, config, save_dir):

    model.eval()
    model.requires_grad_(False)
    model.to(device)

    os.makedirs(save_dir, exist_ok=True)

    # =========================
    # instrumentation
    # =========================
    spike_logger = SpikeTraceLogger(model)
    adc_counter = ADCProxyCounter()

    spike_logger.attach_hooks()

    correct = 0
    total = 0
    all_logits = []

    # =========================
    # inference loop
    # =========================
    with torch.no_grad():

        for data, target in dataloader:

            data, target = data.to(device), target.to(device)

            spike_logger.reset_step()

            output = model(data)

            all_logits.append(output.cpu())

            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

            # =========================
            # STEP instrumentation
            # =========================
            spike_logger.step_finalize()

            adc_counter.update_from_spikes(
                spike_logger.get_step_spikes()
            )

    # =========================
    # global stats
    # =========================
    accuracy = correct / total

    spike_trace = spike_logger.get_full_trace()
    spike_stats = spike_logger.get_statistics()
    adc_stats = adc_counter.finalize()

    # =========================
    # derived hardware signals (VERY IMPORTANT)
    # =========================

    spike_rate = spike_stats["global"]["mean_activity"]
    adc_requests = adc_stats["total_requests"]

    # proxy SOP estimate (for Section 4 model)
    sop_estimate = adc_requests * config.get("fanout", 64)

    # =========================
    # save outputs (Section 4 READY)
    # =========================

    torch.save(spike_trace, os.path.join(save_dir, "spike_trace.pt"))

    np.save(os.path.join(save_dir, "logits.npy"),
            torch.cat(all_logits).numpy())

    with open(os.path.join(save_dir, "accuracy.json"), "w") as f:
        json.dump({"accuracy": accuracy}, f, indent=2)

    with open(os.path.join(save_dir, "spike_activity.json"), "w") as f:
        json.dump({
            "spike_rate": spike_rate,
            "layer": spike_stats["layer"]
        }, f, indent=2)

    with open(os.path.join(save_dir, "adc_proxy.json"), "w") as f:
        json.dump(adc_stats, f, indent=2)

    with open(os.path.join(save_dir, "derived_hardware.json"), "w") as f:
        json.dump({
            "spike_rate": spike_rate,
            "adc_requests": adc_requests,
            "sop_estimate": sop_estimate
        }, f, indent=2)

    with open(os.path.join(save_dir, "eval_meta.json"), "w") as f:
        json.dump({
            "T": config.get("T", 10),
            "dataset": config.get("dataset", ""),
            "note": "HIPSA eval00 instrumentation layer"
        }, f, indent=2)

    print(f"[EVAL00 DONE] Acc={accuracy:.4f} | Spike={spike_rate:.4f} | ADC={adc_requests}")

    return accuracy