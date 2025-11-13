import torch

print(torch.cuda.is_available()) # 检查 CUDA 是否可用
print(torch.cuda.device_count()) # 查看可用 GPU 数量