"""Trace-driven HAPR robustness/latency design-space closure.

The sweep reuses real runtime partial sums and the declared architecture-level
front-end model.  It does not claim transistor-level TIA power or PDK closure.
"""
from __future__ import annotations

import argparse, csv, json, math, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.hapr_statistical_feasibility import run_joint_monte_carlo


def load_trace(path: Path):
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if not rows or any("real_runtime" not in r.get("trace_provenance", "") for r in rows):
        raise ValueError("Only real-runtime HAPR partial-sum traces are accepted")
    return rows


def system_point(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))["selected_operating_point"]
    s, m = d["selected_design"], d["metrics"]
    return dict(latency_us=float(m["latency_us_per_image"]), energy_uj=float(m["energy_uJ_per_image"]),
                power_w=float(m["nominal_power_w"]), transactions=float(s["transactions_issued_per_image"]))


def pareto(rows):
    feasible = [r for r in rows if r["worst_settling_violation_probability"] == 0 and r["worst_saturation_probability"] == 0]
    for r in rows: r["pareto"] = 0
    for a in feasible:
        dominated = any(
            b is not a and b["mean_energy_uJ"] <= a["mean_energy_uJ"] and
            b["mean_latency_us"] <= a["mean_latency_us"] and
            b["worst_capacitance_multiplier_closed"] >= a["worst_capacitance_multiplier_closed"] and
            (b["mean_energy_uJ"] < a["mean_energy_uJ"] or b["mean_latency_us"] < a["mean_latency_us"] or
             b["worst_capacitance_multiplier_closed"] > a["worst_capacitance_multiplier_closed"])
            for b in feasible)
        if not dominated: a["pareto"] = 1
    return [r for r in rows if r["pareto"]]


def main():
    p=argparse.ArgumentParser(); p.add_argument("--trace", required=True); p.add_argument("--device-params", default="configs/device_params.yaml")
    p.add_argument("--cnn-system", required=True); p.add_argument("--transformer-system", required=True); p.add_argument("--output", required=True)
    p.add_argument("--samples", type=int, default=10000); p.add_argument("--seed", type=int, default=42); a=p.parse_args()
    rows=load_trace(Path(a.trace)); dev=yaml.safe_load(Path(a.device_params).read_text(encoding="utf-8")); base=dict(dev["analog_validation"])
    base.update(dict(mrr_sigma=.02,laser_sigma=.01,tia_gain_sigma=.01,branch_mismatch_sigma=.01,pd_responsivity_sigma=.03,
                     adc_full_scale_sigma=.01,settling_error_fraction=1/128))
    systems={"dvsgesture:SpikingGestureCNN":system_point(Path(a.cnn_system)),
             "dvsgesture:SpikingGestureTinyTransformer":system_point(Path(a.transformer_system))}
    grouped={w:[r for r in rows if r["workload"]==w] for w in sorted({r["workload"] for r in rows})}
    details=[]; summary=[]
    for req in (120.,160.,200.,240.):
      for window in (.8,1.0,1.2,1.5):
        point=[]
        for cap in (1.0,1.2,1.25,1.5):
          for workload, wr in grouped.items():
            trace={k:np.asarray([float(r[k]) for r in wr]) for k in ("fan_in","i_pos_a","i_neg_a")}
            cfg=dict(base, settling_equivalent_resistance_ohm=req, settling_time_limit_ns=window, tia_transimpedance_ohm=req)
            q=run_joint_monte_carlo(trace,cfg,samples=a.samples,seed=a.seed,capacitance_multiplier=cap)
            q.update(workload=workload,settling_equivalent_resistance_ohm=req,service_window_ns=window)
            sys=systems[workload]; extra=max(math.ceil(window)-1,0)*sys["transactions"]/4/1000
            q["system_latency_us"]=sys["latency_us"]+extra; q["system_energy_uJ"]=sys["energy_uj"]+sys["power_w"]*extra
            q["energy_model_scope"]="same_declared_frontend_power; conservative non-pipelined extra service cycles; no transistor-level R-dependent TIA power"
            details.append(q); point.append(q)
        cap_closed=max([c for c in (1.0,1.2,1.25,1.5) if all(x["settling_violation_probability"]==0 and x["saturation_probability"]==0 for x in point if x["capacitance_multiplier"]==c)] or [0])
        summary.append(dict(settling_equivalent_resistance_ohm=req,service_window_ns=window,
          worst_capacitance_multiplier_closed=cap_closed,worst_settling_violation_probability=max(x["settling_violation_probability"] for x in point),
          worst_saturation_probability=max(x["saturation_probability"] for x in point),mean_latency_us=np.mean([x["system_latency_us"] for x in point if x["capacitance_multiplier"]==1.5]),
          mean_energy_uJ=np.mean([x["system_energy_uJ"] for x in point if x["capacitance_multiplier"]==1.5]),
          min_adc6_snr_pass_fraction_nonzero=min(x["adc6_snr_pass_fraction_nonzero"] for x in point)))
    front=pareto(summary); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    for name,data in (("design_space_detail.csv",details),("design_space_summary.csv",summary),("pareto_front.csv",front)):
      with (out/name).open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    selected=min((r for r in summary if r["worst_capacitance_multiplier_closed"]>=1.5),key=lambda r:(r["mean_latency_us"],-r["settling_equivalent_resistance_ohm"]))
    (out/"summary.json").write_text(json.dumps({"selected_architecture_point":selected,"pareto_front":front,
      "claim_boundary":"trace-driven architecture model; R-dependent TIA power, extracted parasitics, PVT and stability require circuit validation"},indent=2)+"\n",encoding="utf-8")
    fig,ax=plt.subplots(figsize=(6.2,4.2));
    for r in summary:
      ax.scatter(r["mean_latency_us"],r["mean_energy_uJ"],c=r["worst_capacitance_multiplier_closed"],vmin=1,vmax=1.5,cmap="viridis",s=35 if not r["pareto"] else 90,marker="o" if not r["pareto"] else "*")
    ax.set(xlabel="Mean end-to-end latency (us)",ylabel="Mean end-to-end energy (uJ)",title="HAPR robustness-latency-energy design space");fig.tight_layout();fig.savefig(out/"fig_hapr_pareto.png",dpi=300);fig.savefig(out/"fig_hapr_pareto.pdf");plt.close(fig)
    print(json.dumps({"selected":selected,"pareto_points":len(front)},indent=2))
if __name__=="__main__": main()
