import torch
import pandas as pd
import numpy as np
import binascii

# 1. 预处理函数 (Hex -> Tensor)
def hex_to_tensor(hex_str, max_len=256):
    try:
        if pd.isna(hex_str): return torch.zeros(max_len, dtype=torch.long)
        s = str(hex_str).strip()
        if len(s) == 0: return torch.zeros(max_len, dtype=torch.long)
        if len(s) % 2 != 0: s = s + '0'  # 在末尾补0，而不是开头
        indices = list(binascii.unhexlify(s))
        if len(indices) > max_len:
            indices = indices[:max_len]
        else:
            indices = indices + [0] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)
    except:
        return torch.zeros(max_len, dtype=torch.long)

def analyze_hamming_distances(csv_path, max_len=256, max_samples_per_class=200, device='cuda:0'):
    print("正在加载数据...")
    df = pd.read_csv(csv_path)
    
    # 检查必要的列是否存在
    if 'label' not in df.columns:
        print("错误: CSV文件中没有找到'label'列")
        return
    if 'payload' not in df.columns:
        print("错误: CSV文件中没有找到'payload'列")
        return
    
    # 过滤掉空标签和空payload
    df = df.dropna(subset=['label', 'payload'])
    df = df[df['label'].astype(str) != 'nan']
    
    print(f"加载了 {len(df)} 条有效数据")
    
    # 转换为 Tensor 列表
    # 结构: { 'label_A': Tensor(N, 256), 'label_B': Tensor(M, 256) ... }
    class_payloads = {}
    
    print("正在预处理 Payload...")
    # 按标签分组
    grouped = df.groupby('label')
    for label, group in grouped:
        # 策略 B: 限制采样，避免头部类别计算量爆炸
        if len(group) > max_samples_per_class:
            group = group.sample(max_samples_per_class, random_state=42)
        
        # 过滤掉无效的payload
        valid_payloads = []
        for p in group['payload']:
            tensor = hex_to_tensor(p, max_len)
            if tensor.sum() > 0:  # 至少有一些非零值
                valid_payloads.append(tensor)
        
        if len(valid_payloads) > 0:
            class_payloads[str(label)] = torch.stack(valid_payloads).float() # 用 float 方便后续矩阵运算(虽然是int)
            print(f"  标签 '{label}': {len(valid_payloads)} 个有效样本")
    
    print(f"共找到 {len(class_payloads)} 个有效类别")
    
    # 使用指定的设备
    torch_device = torch.device(device)
    if not torch.cuda.is_available() and 'cuda' in str(torch_device):
        print(f"警告: CUDA不可用，改用CPU")
        torch_device = torch.device('cpu')
    results = {'intra': [], 'inter': []}

    print(f"开始计算 (使用设备: {torch_device})...")
    
    # 获取所有类别列表
    labels = list(class_payloads.keys())
    
    with torch.no_grad():
        # --- 1. 计算类内距离 (Intra-class) ---
        for label in labels:
            tensor = class_payloads[label].to(torch_device) # (N, L)
            n = tensor.shape[0]
            if n < 2: continue
            
            # 策略 A: 矩阵广播计算
            # (N, 1, L) != (1, N, L) -> (N, N, L) -> sum(-1) -> (N, N)
            # 这里的 != 实际上就是 Hamming Distance (不相等的字节数)
            diff_matrix = (tensor.unsqueeze(1) != tensor.unsqueeze(0)).sum(dim=-1).float()
            
            # 取上三角矩阵（不含对角线），避免重复计算和自对比
            # tril_indices 得到下三角索引，设置为 NaN 或忽略
            mask = torch.triu(torch.ones(n, n), diagonal=1).bool().to(torch_device)
            distances = diff_matrix[mask] # 只取上三角部分的有效距离
            
            avg_dist = distances.mean().item()
            results['intra'].append(avg_dist)
            # print(f"[类内] {label}: 平均汉明距离 = {avg_dist:.2f}")

        # --- 2. 计算类间距离 (Inter-class) ---
        # 为了演示，这里只随机抽样部分类对，或者计算所有类对
        # 策略 C 的变体: 依然使用 Tensor 计算，但因为 N 较小(被Cap限制了)，可以直接两两算
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                label_a = labels[i]
                label_b = labels[j]
                
                tensor_a = class_payloads[label_a].to(torch_device) # (N, L)
                tensor_b = class_payloads[label_b].to(torch_device) # (M, L)
                
                # 广播计算: (N, 1, L) vs (1, M, L) -> (N, M) 距离矩阵
                dists = (tensor_a.unsqueeze(1) != tensor_b.unsqueeze(0)).sum(dim=-1).float()
                
                avg_inter_dist = dists.mean().item()
                results['inter'].append(avg_inter_dist)
                
    print("\n=== 统计结果 ===")
    if len(results['intra']) > 0 and len(results['inter']) > 0:
        avg_intra = np.mean(results['intra'])
        avg_inter = np.mean(results['inter'])
        print(f"平均类内距离 (Intra): {avg_intra:.2f}")
        print(f"平均类间距离 (Inter): {avg_inter:.2f}")
        print(f"类内距离标准差: {np.std(results['intra']):.2f}")
        print(f"类间距离标准差: {np.std(results['inter']):.2f}")
        print(f"类内距离范围: [{np.min(results['intra']):.2f}, {np.max(results['intra']):.2f}]")
        print(f"类间距离范围: [{np.min(results['inter']):.2f}, {np.max(results['inter']):.2f}]")
        
        if avg_intra < avg_inter:
            print("\n✅ 验证成功：同类样本更相似，检索逻辑成立！")
            print(f"   类间距离比类内距离大 {((avg_inter - avg_intra) / avg_intra * 100):.2f}%")
        else:
            print("\n⚠️ 警告：类间类内区分度不明显，可能需要更强的 Payload 特征提取器（如训练后的 Encoder）。")
    else:
        print("错误: 没有足够的数据进行计算")

if __name__ == "__main__":
    # 使用 train.csv 和 cuda:0
    csv_path = "data/train.csv"
    print(f"分析文件: {csv_path}")
    print("=" * 60)
    analyze_hamming_distances(csv_path, max_len=256, max_samples_per_class=200, device='cuda:0')