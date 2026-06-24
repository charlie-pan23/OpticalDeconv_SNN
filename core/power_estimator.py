class PowerEstimator:
    """
    HIPSA 体系结构级功耗评估引擎 (Architecture Power Evaluation Engine)
    功能：基于物理常数表、稀疏度、ADC触发率，严密推导组件级功耗。
    """

    def __init__(self, config_yaml):
        """传入解析好的 yaml 配置字典"""
        self.hw = config_yaml['hardware']
        self.pm = config_yaml['power_model']

        # 提取论文标称设定的基准参数 (Baseline Config)
        self.baseline_active_ratio = 0.15  # Main case 基准神经元活性 (15%)
        self.baseline_adc_activity = 0.38  # Main case 基准 ADC 激活率 (38%)

    def compute_power_breakdown(self, actual_active_ratio, actual_adc_activity):
        """
        核心物理公式计算函数
        :param actual_active_ratio: eval_01 跑出来的真实神经元脉冲率
        :param actual_adc_activity: HAPR 仿真跑出来的真实 ADC 触发率
        """
        # 1. 静态功耗 (无论数据多稀疏，只要激光器和MRR不关断，这部分功耗雷打不动)
        static_laser_w = self.pm["cw_laser_source_mw"] / 1000.0
        static_mrr_w = self.pm["global_mrr_stabilization_mw"] / 1000.0
        static_leakage_w = self.pm["leakage_misc_io_mw"] / 1000.0

        total_static_w = static_laser_w + static_mrr_w + static_leakage_w

        # 2. 动态功耗 (严格按照事件驱动电路的 Activity Factor 进行等比例缩放)
        # SNN/电学接口动态功耗缩放因子 (α / α_baseline)
        spike_scaling = actual_active_ratio / self.baseline_active_ratio

        dyn_modulator_w = (self.pm["event_gated_modulator_drivers_mw"] / 1000.0) * spike_scaling
        dyn_sram_w = (self.pm["sram_register_files_mw"] / 1000.0) * spike_scaling
        dyn_noc_w = (self.pm["noc_bus_controller_clock_mw"] / 1000.0) * spike_scaling
        dyn_pd_tia_w = (self.pm["pd_tia_comparator_mw"] / 1000.0) * spike_scaling

        # ADC 功耗单独按照模拟器统计的 ADC Activity 进行缩放
        adc_scaling = actual_adc_activity / self.baseline_adc_activity
        dyn_adc_w = (self.pm["shared_adc_pool_mw"] / 1000.0) * adc_scaling

        total_dynamic_w = dyn_modulator_w + dyn_sram_w + dyn_noc_w + dyn_pd_tia_w + dyn_adc_w
        total_system_w = total_static_w + total_dynamic_w

        return {
            "static_w": total_static_w,
            "dynamic_w": total_dynamic_w,
            "total_w": total_system_w,
            "breakdown": {
                "laser": static_laser_w,
                "mrr": static_mrr_w,
                "modulator": dyn_modulator_w,
                "adc": dyn_adc_w,
                "sram": dyn_sram_w
            }
        }