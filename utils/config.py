import os
import yaml


def load_config(config_file="config.yaml"):
    """
    加载项目根目录下的 YAML 配置文件，并返回一个 Python 字典。
    """
    # 动态获取根目录路径，确保在哪里运行都能找到 config.yaml
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    config_path = os.path.join(project_root, config_file)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


# 暴露一个全局的 config 变量供其他模块直接导入
config = load_config()

if __name__ == "__main__":
    # 测试读取
    print("--- 配置文件加载测试 ---")
    print(f"训练 Batch Size: {config['training']['batch_size']}")
    print(f"网络 Kernel Size: {config['network']['kernel_size']}")
    print(f"光子阵列数量: {config['hardware']['num_arrays']}")