import os
import numpy as np
import matplotlib.pyplot as plt

def calc_metrics_by_algo(base_dir, skip_first_line=True):
    """
    按算法目录统计平均：
    - Bitrate（第2列）
    - Reward（第8列）
    - Rebuffering（第4列）
    - Bitrate Variation（第7列）
    """
    stats = {}
    for algo_dir in os.listdir(base_dir):
        algo_path = os.path.join(base_dir, algo_dir)
        if not os.path.isdir(algo_path):
            continue

        # 遍历子目录（如 seed_100003 或 early_stop_...）
        for seed_dir in os.listdir(algo_path):
            seed_path = os.path.join(algo_path, seed_dir)
            if not os.path.isdir(seed_path):
                continue

            bitrate_list = []
            reward_list = []
            rebuf_list = []
            smoothness_list = []
            file_count = 0

            for log_file in os.listdir(seed_path):
                file_path = os.path.join(seed_path, log_file)
                if not os.path.isfile(file_path):
                    continue

                file_count += 1
                first_line = True
                with open(file_path, 'r') as f:
                    for line in f:
                        parse = line.strip().split('\t')
                        if len(parse) <= 7:
                            continue
                        if first_line:
                            first_line = False
                            if skip_first_line:
                                continue
                        try:
                            bitrate_list.append(float(parse[1]))
                            rebuf_list.append(float(parse[3]))
                            smoothness_list.append(float(parse[6]))  # Bitrate Variation
                            reward_list.append(float(parse[7]))
                        except ValueError:
                            continue

            stats[algo_dir] = {
                'mean_bitrate': np.mean(bitrate_list) if bitrate_list else 0.0,
                'mean_reward': np.mean(reward_list) if reward_list else 0.0,
                'mean_rebuffering': np.mean(rebuf_list) if rebuf_list else 0.0,
                'mean_bitrate_variation': np.mean(smoothness_list) if smoothness_list else 0.0,
                'files_processed': file_count
            }

    return stats


def plot_metrics(stats):
    """
    stats: dict, {algo_name: {'mean_bitrate':..., 'mean_reward':..., 'mean_rebuffering':..., 'mean_bitrate_variation':...}}
    一次性绘制四个子图，展示四个指标对比。
    """
    algos = list(stats.keys())

    mean_bitrate = [stats[a]['mean_bitrate'] for a in algos]
    mean_reward = [stats[a]['mean_reward'] for a in algos]
    mean_rebuffering = [stats[a]['mean_rebuffering'] for a in algos]
    mean_bitrate_variation = [stats[a]['mean_bitrate_variation'] for a in algos]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    metrics = [
        ('Average Bitrate', mean_bitrate),
        ('Average Reward', mean_reward),
        ('Average Rebuffering', mean_rebuffering),
        ('Average Bitrate Variation', mean_bitrate_variation)
    ]

    colors = ['skyblue', 'salmon', 'lightgreen', 'orange']

    for ax, (metric_name, values) in zip(axes, metrics):
        ax.bar(algos, values, color=colors)
        ax.set_title(metric_name)
        ax.set_ylabel(metric_name)

        # 设置 y 轴上限，防止数字被裁掉
        top_margin = max(values) * 1.1
        ax.set_ylim(0, top_margin)

        for i, v in enumerate(values):
            ax.text(i, v + 0.01*max(values), f"{v:.4f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()



# -------------------- 使用示例 --------------------
base_dir = "artifacts/results/fcc-test_video1/trace_num_100_fixed_True"

# 统计指标
stats = calc_metrics_by_algo(base_dir)

# 打印结果
for algo, s in stats.items():
    print(f"Algorithm: {algo}")
    print(f"  Files processed: {s['files_processed']}")
    print(f"  Average Bitrate: {s['mean_bitrate']:.2f}")
    print(f"  Average Reward: {s['mean_reward']:.2f}")
    print(f"  Average Rebuffering: {s['mean_rebuffering']:.2f}")
    print(f"  Average Bitrate Variation: {s['mean_bitrate_variation']:.2f}")

# 绘制图表
plot_metrics(stats)
