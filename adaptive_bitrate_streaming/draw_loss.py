import re
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免图形界面问题
import matplotlib.pyplot as plt

# log_file = "console.log"  # 你的训练日志文件路径
log_file = "console.log"

# 正则表达式
# 匹配格式：Training Iteration #数字 ... 'training/train_loss_mean': np.float64(数字)
loss_pattern = re.compile(r"Training Iteration #(\d+).*?'training/train_loss_mean':\s*np\.float64\(([0-9.]+)\)", re.S)
# 匹配格式：Evaluation Information ... 'episodes_return': np.float64(数字)
return_pattern = re.compile(r"Evaluation Information.*?'episodes_return':\s*np\.float64\(([0-9.]+)\)", re.S)

with open(log_file, "r", encoding="utf-8") as f:
    text = f.read()

# 解析 loss
loss_data = loss_pattern.findall(text)
# 解析 return
return_data = return_pattern.findall(text)

loss_data = [(int(step), float(loss)) for step, loss in loss_data]
return_data = [float(r) for r in return_data]

# 打印解析结果用于验证
print(f"解析到 {len(loss_data)} 个 loss 数据点")
print(f"前5个 loss 数据: {loss_data[:5]}")
print(f"解析到 {len(return_data)} 个 return 数据点")
print(f"前5个 return 数据: {return_data[:5]}")

# -------------------- 数据处理 --------------------

# Train Loss：每两个 iteration 采样 (保持不变)
loss_steps = [step for step, _ in loss_data]
loss_vals = [loss for _, loss in loss_data]

# Evaluation Return：直接使用所有数据，因为你说明了 return 已经是每两个训练迭代才有一次
# 评估步骤的索引应该乘以 2，来对应实际的训练迭代次数
# e.g., 第 0 个 return 对应 0 迭代, 第 1 个 return 对应 2 迭代, 第 2 个 return 对应 4 迭代...
return_vals = return_data
return_steps = [i * 2 for i in range(len(return_vals))]

# ----------------- 画图 -----------------

## 📉 Train Loss 曲线
plt.figure(figsize=(10, 4))
plt.plot(loss_steps, loss_vals, marker='o', linestyle='-', color='tab:blue', markersize=3)
plt.title("Train Loss")
plt.xlabel("Training Iteration")
plt.ylabel("Loss Mean")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("train_loss.png", dpi=150, bbox_inches='tight')
print("Train Loss 图已保存为 train_loss.png")
plt.close()

## 📈 Evaluation Return 曲线
plt.figure(figsize=(10, 4))
plt.plot(return_steps, return_vals, marker='s', linestyle='-', color='tab:orange', markersize=3)
plt.title("Evaluation Return")
plt.xlabel("Training Iteration")
plt.ylabel("Evaluation Return")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("evaluation_return.png", dpi=150, bbox_inches='tight')
print("Evaluation Return 图已保存为 evaluation_return.png")
plt.close()