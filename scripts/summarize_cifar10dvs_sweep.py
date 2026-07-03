from pathlib import Path
import csv
import re

ROOT = Path("results/eval_v2_local_sweep/cifar10dvs")
OUT = ROOT / "cifar10dvs_batch_sweep_energy_summary.csv"

rows = []

for csv_path in sorted(ROOT.glob("*/cifar10dvs/eval_06/batch_sweep_summary.csv")):
    run_name = csv_path.parts[-4]  # cpu_b1 / gpu_b32

    m = re.match(r"(cpu|gpu)_b(\d+)", run_name)
    if not m:
        print(f"[skip] unexpected run folder: {run_name}")
        continue

    expected_device = "cuda" if m.group(1) == "gpu" else "cpu"
    expected_batch = int(m.group(2))

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            device = r.get("device", "")
            batch_size = int(float(r.get("batch_size", -1)))

            # 每个文件理论上只有一个 batch，但保险起见过滤一下
            if device != expected_device:
                continue
            if batch_size != expected_batch:
                continue

            latency = float(r["mean_latency_ms_per_image"])
            throughput = float(r["throughput_images_per_s"])
            active_power = float(r["active_power_w"]) if r.get("active_power_w") else None
            energy = float(r["energy_mJ_per_image"]) if r.get("energy_mJ_per_image") else None

            rows.append({
                "run": run_name,
                "device": device,
                "batch_size": batch_size,
                "mean_latency_ms_per_image": latency,
                "throughput_images_per_s": throughput,
                "active_power_w": active_power,
                "energy_mJ_per_image": energy,
                "num_timed_batches": r.get("num_timed_batches", ""),
                "total_timed_samples": r.get("total_timed_samples", ""),
                "source": str(csv_path),
            })

rows.sort(key=lambda x: (0 if x["device"] == "cpu" else 1, x["batch_size"]))

if not rows:
    raise SystemExit(f"No summary rows found under {ROOT}")

fieldnames = [
    "run",
    "device",
    "batch_size",
    "mean_latency_ms_per_image",
    "throughput_images_per_s",
    "active_power_w",
    "energy_mJ_per_image",
    "num_timed_batches",
    "total_timed_samples",
    "source",
]

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"[ok] wrote: {OUT}")
print()
print("device,batch,latency_ms_per_image,throughput_img_s,active_power_w,energy_mJ_per_image")
for r in rows:
    print(
        f"{r['device']},{r['batch_size']},"
        f"{r['mean_latency_ms_per_image']:.6f},"
        f"{r['throughput_images_per_s']:.3f},"
        f"{r['active_power_w']:.3f},"
        f"{r['energy_mJ_per_image']:.6f}"
    )

# 选出 CPU/GPU 各自 energy 最低和 latency 最低的 batch
print()
for device in ["cpu", "cuda"]:
    dev_rows = [r for r in rows if r["device"] == device]
    if not dev_rows:
        continue

    best_latency = min(dev_rows, key=lambda x: x["mean_latency_ms_per_image"])
    best_energy = min(dev_rows, key=lambda x: x["energy_mJ_per_image"])

    print(f"[{device}] best latency: batch={best_latency['batch_size']}, "
          f"{best_latency['mean_latency_ms_per_image']:.6f} ms/image, "
          f"{best_latency['throughput_images_per_s']:.3f} img/s")

    print(f"[{device}] best energy : batch={best_energy['batch_size']}, "
          f"{best_energy['energy_mJ_per_image']:.6f} mJ/image, "
          f"active_power={best_energy['active_power_w']:.3f} W")