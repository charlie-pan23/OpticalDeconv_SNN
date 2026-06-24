import os
import yaml
import json
from utils.Logger import logger


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    logger.info("=== Phase 2: Architecture Power & Performance Estimation ===")

    # 默认读取主基准配置
    config_path = "configs/config_cifar10dvs.yaml"
    stats_path = "results/activity_stats.json"

    if not os.path.exists(stats_path):
        logger.error(f"Activity stats not found at {stats_path}! Please run eval_01_activity.py first.")
        return

    config = load_config(config_path)
    with open(stats_path, 'r') as f:
        stats = json.load(f)

    active_ratio = stats["active_ratio"]
    adc_activity = stats["adc_activity"]
    network_type = stats["network"]

    hw = config["hardware"]
    pm = config["power_model"]

    # ---------------------------------------------------------
    # 1. 性能推导 (Performance Derivation) - 对应 Table 4
    # ---------------------------------------------------------
    # 依据网络结构给定标称的 Dense-equivalent SOPs per image (基于 T=10)
    if network_type == "snn_vgg":
        dense_equivalent_sops_per_image = 3.13e9  # 论文 Main Case 的标称值
    else:
        # DVS Gesture 的浅层 CNN 计算量较小，此处为估算示例值
        dense_equivalent_sops_per_image = 4.5e8

    actual_sops_per_image = dense_equivalent_sops_per_image * active_ratio

    # 吞吐与延迟推导
    realized_processing_rate = hw["realized_processing_rate"] * 1e12  # 6.55 TSOP/s
    latency_seconds = actual_sops_per_image / realized_processing_rate
    throughput_fps = 1.0 / latency_seconds

    logger.info("--- Table 4: HIPSA Main Performance ---")
    logger.info(f"Dense-equiv. SOP/image : {dense_equivalent_sops_per_image:.2e}")
    logger.info(f"Extracted Active Ratio : {active_ratio * 100:.2f}%")
    logger.info(f"Actual SOP/image       : {actual_sops_per_image:.2e}")
    logger.info(f"Realized Process Rate  : {hw['realized_processing_rate']} TSOP/s")
    logger.info(f"Latency/image          : {latency_seconds * 1e6:.1f} μs")
    logger.info(f"Throughput             : {throughput_fps:,.0f} image/s")

    # ---------------------------------------------------------
    # 2. 功耗推导 (Power Model Derivation) - 对应 Table 3
    # ---------------------------------------------------------
    # 静态功耗：不随稀疏度改变 (TS-EGI 创新的核心体现：光路热稳定)
    static_power_w = (pm["cw_laser_source_mw"] +
                      pm["global_mrr_stabilization_mw"] +
                      pm["leakage_misc_io_mw"]) / 1000.0

    # 动态功耗自适应缩放机制 (Dynamic Power Scaling)
    # yaml 中记录的是基于 Main Case (约 15% Active Ratio) 算出的标称值
    # 如果实际跑出来的 active_ratio 更低，动态功耗将等比例下降
    baseline_active_ratio = 0.15
    sparsity_scaling_factor = active_ratio / baseline_active_ratio

    dynamic_power_w = (pm["event_gated_modulator_drivers_mw"] +
                       pm["pd_tia_comparator_mw"] +
                       pm["shared_adc_pool_mw"] +
                       pm["sram_register_files_mw"] +
                       pm["noc_bus_controller_clock_mw"]) / 1000.0

    # 施加稀疏缩放因子
    scaled_dynamic_power_w = dynamic_power_w * sparsity_scaling_factor
    total_power_w = static_power_w + scaled_dynamic_power_w

    # 算总能耗与能效比
    energy_per_image_uj = total_power_w * latency_seconds * 1e6
    actual_efficiency = (actual_sops_per_image / (energy_per_image_uj * 1e-6)) / 1e12
    dense_efficiency = (dense_equivalent_sops_per_image / (energy_per_image_uj * 1e-6)) / 1e12

    logger.info("\n--- Component-Level Power Model ---")
    logger.info(f"Static Power  (Laser/MRR/Leakage) : {static_power_w * 1000:.1f} mW")
    logger.info(
        f"Dynamic Power (Scaled by Sparsity): {scaled_dynamic_power_w * 1000:.1f} mW (Scaling factor: {sparsity_scaling_factor:.2f}x)")
    logger.info(f"Total System Power                : {total_power_w:.2f} W")

    logger.info("\n--- Final Efficiency Metrics (Table 4) ---")
    logger.info(f"Energy/image                : {energy_per_image_uj:.1f} μJ")
    logger.info(f"Actual Efficiency           : {actual_efficiency:.2f} TSOPS/W")
    logger.info(f"Dense-equivalent Efficiency : {dense_efficiency:.2f} TSOPS/W")
    logger.info("=========================================================")


if __name__ == "__main__":
    main()