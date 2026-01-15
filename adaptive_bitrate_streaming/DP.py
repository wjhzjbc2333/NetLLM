import os
import glob
import pickle
import numpy as np

# ================= 配置参数 =================
BITS_IN_BYTE = 8.0
B_IN_MB = 1000000.0
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # ms
MILLISECONDS_IN_SECOND = 1000.0
VIDEO_CHUNK_LEN = 4000.0  # ms

# QOE 参数
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BITRATE_LEVELS = len(VIDEO_BIT_RATE)
M_IN_K = 1000.0
REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1.0


# ================= 数据加载函数 =================

def load_single_trace(file_path):
    times = []
    bws = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) < 2: continue
            try:
                t = float(parts[0])
                bw = float(parts[1])
                times.append(t)
                bws.append(bw)
            except ValueError:
                continue
    return times, bws


def load_video_sizes(video_size_dir):
    video_size = {}
    for bitrate in range(BITRATE_LEVELS):
        fname = f"video_size_{bitrate}"
        file_path = os.path.join(video_size_dir, fname)
        sizes = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    sizes.append(int(line))
        video_size[bitrate] = sizes
    return video_size


# ================= DP 求解逻辑 =================

class State:
    """DP 状态节点"""

    def __init__(self, reward, ptr, last_time, buffer_size, history, total_rebuf):
        self.reward = reward  # 累计 QoE
        self.ptr = ptr  # 当前 trace 行号索引
        self.last_time = last_time  # 累计消耗的绝对时间 (trace 中的时间)
        self.buffer_size = buffer_size  # 当前缓冲区 (ms)
        self.history = history  # 决策路径 list[int]
        self.total_rebuf = total_rebuf  # 统计用


def solve_optimal_abr(times, bws, video_size, buffer_quantization_ms=100.0):
    """
    使用动态规划计算最优 ABR 路径
    """
    total_chunks = len(video_size[0])

    # --- 1. 初始化 ---
    # current_states key: (last_bitrate, buffer_bin_index)
    current_states = {}

    # 初始状态:
    #   last_bitrate=0 (假设开始前是最低画质，或者你可以设为 None 特殊处理，这里为了简化平滑惩罚设为 0)
    #   buffer=0
    #   ptr=1 (模拟器通常从第2行开始读取，第1行作为初始时间参考)
    #   last_time=times[0]

    init_state = State(
        reward=0.0,
        ptr=1,
        last_time=times[0],
        buffer_size=0.0,
        history=[],
        total_rebuf=0.0
    )

    # Key = (last_bitrate_idx, buffer_bin)
    current_states[(0, 0)] = init_state

    print(f"Start DP solving for {total_chunks} chunks...")

    # --- 2. 逐层 DP (Chunk by Chunk) ---
    for chunk_idx in range(total_chunks):
        next_states = {}

        # 遍历上一层所有存活状态
        for (prev_q, _), curr_state in current_states.items():

            # 尝试当前 chunk 的所有码率决策
            for next_q in range(BITRATE_LEVELS):

                # === A. 模拟环境下载 (Physics Simulation) ===
                chunk_size_bytes = video_size[next_q][chunk_idx]

                # 复制当前模拟器状态
                ptr = curr_state.ptr
                last_time = curr_state.last_time
                buffer_size = curr_state.buffer_size

                downloaded = 0.0
                delay = 0.0  # seconds

                # 模拟下载过程 (Trace Simulation)
                curr_ptr_temp = ptr
                curr_last_time_temp = last_time

                while True:
                    bw = bws[curr_ptr_temp]  # Mbps
                    throughput = bw * B_IN_MB / BITS_IN_BYTE  # B/s
                    duration = times[curr_ptr_temp] - curr_last_time_temp

                    # 异常处理：Trace 耗尽循环
                    if duration <= 0:
                        curr_ptr_temp += 1
                        if curr_ptr_temp >= len(bws):
                            curr_ptr_temp = 1
                            curr_last_time_temp = 0.0
                        continue

                    payload = throughput * duration * PACKET_PAYLOAD_PORTION

                    if downloaded + payload >= chunk_size_bytes:
                        remain = chunk_size_bytes - downloaded
                        fractional_time = remain / (throughput * PACKET_PAYLOAD_PORTION)
                        delay += fractional_time
                        curr_last_time_temp += fractional_time
                        break
                    else:
                        downloaded += payload
                        delay += duration
                        curr_last_time_temp = times[curr_ptr_temp]
                        curr_ptr_temp += 1

                        # Trace 循环机制
                        if curr_ptr_temp >= len(bws):
                            curr_ptr_temp = 1
                            curr_last_time_temp = 0.0

                # 计算延迟、卡顿、新缓冲区
                delay_ms = delay * MILLISECONDS_IN_SECOND + LINK_RTT
                rebuf_ms = max(delay_ms - buffer_size, 0.0)
                new_buffer_size = max(buffer_size - delay_ms, 0.0) + VIDEO_CHUNK_LEN

                # === B. 计算 QoE Reward ===
                # 1. Video Quality Reward
                r_quality = VIDEO_BIT_RATE[next_q] / M_IN_K

                # 2. Rebuffer Penalty
                r_rebuf = REBUF_PENALTY * (rebuf_ms / MILLISECONDS_IN_SECOND)

                # 3. Smoothness Penalty
                # 如果是第一个 chunk，没有过去码率，惩罚为 0
                if chunk_idx == 0:
                    r_smooth = 0.0
                else:
                    r_smooth = SMOOTH_PENALTY * abs(VIDEO_BIT_RATE[next_q] - VIDEO_BIT_RATE[prev_q]) / M_IN_K

                step_reward = r_quality - r_rebuf - r_smooth
                total_reward = curr_state.reward + step_reward

                # === C. 状态合并与剪枝 ===
                # 限制 buffer 上限以减少状态空间 (60秒)
                new_buffer_size_capped = min(new_buffer_size, 60000.0)

                # 离散化 buffer: 100ms 一个桶
                buffer_bin = int(new_buffer_size_capped / buffer_quantization_ms)

                state_key = (next_q, buffer_bin)

                # 核心 DP 逻辑：如果到达相同状态 (Quality, Buffer)，只保留 Reward 更高的
                if state_key not in next_states or total_reward > next_states[state_key].reward:
                    next_states[state_key] = State(
                        reward=total_reward,
                        ptr=curr_ptr_temp,
                        last_time=curr_last_time_temp,
                        buffer_size=new_buffer_size_capped,
                        history=curr_state.history + [next_q],
                        total_rebuf=curr_state.total_rebuf + (rebuf_ms / 1000.0)
                    )

        current_states = next_states
        # 打印日志
        if (chunk_idx + 1) % 10 == 0:
            print(f"  Processed {chunk_idx + 1}/{total_chunks} chunks. States: {len(current_states)}")

    # --- 3. 结果回溯 ---
    best_final_state = None
    max_reward = -float('inf')

    for s in current_states.values():
        if s.reward > max_reward:
            max_reward = s.reward
            best_final_state = s

    return best_final_state

# ================= 生成 Oracle 经验池 =================

def generate_oracle_exp_pool(
    trace_folder,
    video_size_dir,
    output_path,
    trace_limit: int = -1,
    seed: int = 100003,
):
    """
    使用 DP Oracle 在给定的 trace 文件夹和视频大小目录上生成经验池（ExperiencePool）。

    生成的状态 / 动作 / 奖励 / done 标记与 `generate_exp_pool.py` 中一致：
      - state: 形状为 (S_INFO, S_LEN) 的 numpy 数组
      - action: 码率等级索引 0..BITRATE_LEVELS-1
      - reward: 标量 QoE（quality - rebuffer_penalty - smoothness_penalty）
      - done: 是否为该 episode 的最后一个 chunk

    Args:
        trace_folder: 存放单条 trace 文件的目录（每个文件是一条带宽时间序列）
        video_size_dir: 存放 video_size_0, video_size_1, ... 的目录
        output_path: 经验池保存路径（.pkl）
        trace_limit: 使用的 trace 数量上限；-1 表示使用该目录下所有 trace
        seed: 随机种子（主要用于可复现性）

    Returns:
        exp_pool: 填充好的 ExperiencePool 实例（同时也会被保存到 output_path）
    """
    from plm_special.data.exp_pool import ExperiencePool
    from baseline_special.utils.constants import (
        S_INFO,
        S_LEN,
        BUFFER_NORM_FACTOR,
        CHUNK_TIL_VIDEO_END_CAP,
    )

    np.random.seed(seed)

    # 加载视频大小信息
    video_size = load_video_sizes(video_size_dir)

    # 收集 trace 文件
    trace_files = sorted(glob.glob(os.path.join(trace_folder, '*')))
    if trace_limit is not None and trace_limit != -1:
        trace_files = trace_files[:trace_limit]

    exp_pool = ExperiencePool()

    for trace_idx, trace_path in enumerate(trace_files):
        times, bws = load_single_trace(trace_path)
        if len(times) == 0 or len(bws) == 0:
            continue

        print(f'[{trace_idx + 1}/{len(trace_files)}] Solving DP for trace: {os.path.basename(trace_path)}')
        best_state = solve_optimal_abr(times, bws, video_size)

        if best_state is None or len(best_state.history) == 0:
            print('  DP returned empty history, skip.')
            continue

        actions = best_state.history  # list[int], 每个元素为码率等级索引
        total_chunks = len(actions)

        # 初始化“物理环境”状态，与 solve_optimal_abr 中保持一致
        ptr = 1
        last_time = times[0]
        buffer_size_ms = 0.0

        # 状态表示，形状 (S_INFO, S_LEN)，与 generate_exp_pool 保持一致
        state = np.zeros((S_INFO, S_LEN), dtype=np.float32)
        last_bit_rate_idx = 0  # 对应 VIDEO_BIT_RATE[0]

        step_rewards = []

        for chunk_idx, action_idx in enumerate(actions):
            curr_bitrate_kbps = VIDEO_BIT_RATE[action_idx]
            video_chunk_size = video_size[action_idx][chunk_idx]

            # ========== 物理下载仿真（与 DP 中相同） ==========
            downloaded = 0.0
            delay = 0.0  # seconds
            curr_ptr = ptr
            curr_last_time = last_time

            while True:
                bw = bws[curr_ptr]  # Mbps
                throughput = bw * B_IN_MB / BITS_IN_BYTE  # B/s
                duration = times[curr_ptr] - curr_last_time

                if duration <= 0:
                    curr_ptr += 1
                    if curr_ptr >= len(bws):
                        curr_ptr = 1
                        curr_last_time = 0.0
                    continue

                payload = throughput * duration * PACKET_PAYLOAD_PORTION

                if downloaded + payload >= video_chunk_size:
                    remain = video_chunk_size - downloaded
                    fractional_time = remain / (throughput * PACKET_PAYLOAD_PORTION)
                    delay += fractional_time
                    curr_last_time += fractional_time
                    break
                else:
                    downloaded += payload
                    delay += duration
                    curr_last_time = times[curr_ptr]
                    curr_ptr += 1

                    if curr_ptr >= len(bws):
                        curr_ptr = 1
                        curr_last_time = 0.0

            # 更新 trace 指针和时间
            ptr = curr_ptr
            last_time = curr_last_time

            # 计算时延 / 卡顿 / 缓冲，与 DP 逻辑一致
            delay_ms = delay * MILLISECONDS_IN_SECOND + LINK_RTT
            rebuf_ms = max(delay_ms - buffer_size_ms, 0.0)
            rebuf_sec = rebuf_ms / MILLISECONDS_IN_SECOND

            buffer_size_ms = max(buffer_size_ms - delay_ms, 0.0) + VIDEO_CHUNK_LEN
            buffer_size_sec = buffer_size_ms / MILLISECONDS_IN_SECOND

            # ========== 计算 QoE Reward ==========
            r_quality = curr_bitrate_kbps / M_IN_K
            r_rebuf = REBUF_PENALTY * rebuf_sec

            if chunk_idx == 0:
                smoothness_diff = 0.0
            else:
                smoothness_diff = abs(
                    curr_bitrate_kbps - VIDEO_BIT_RATE[last_bit_rate_idx]
                ) / M_IN_K

            r_smooth = SMOOTH_PENALTY * smoothness_diff
            reward = r_quality - r_rebuf - r_smooth
            step_rewards.append(reward)

            # ========== 更新状态表示 (S_INFO, S_LEN) ==========
            # 滚动历史
            state = np.roll(state, -1, axis=1)

            # 1) 上一段码率（归一化到 [0,1]）
            state[0, -1] = curr_bitrate_kbps / float(max(VIDEO_BIT_RATE))

            # 2) 当前 buffer（秒），再按 BUFFER_NORM_FACTOR 归一化
            state[1, -1] = buffer_size_sec / BUFFER_NORM_FACTOR

            # 3) 过去带宽测量：chunk_size / 下载时间（kilo byte / ms）
            #    这里使用除去 RTT 的“纯下载时间”近似（与 generate_exp_pool 的实现保持一致思路）
            download_ms = max(delay_ms - LINK_RTT, 1e-6)
            state[2, -1] = float(video_chunk_size) / download_ms / M_IN_K

            # 4) 过去下载时延（ms -> 归一化）
            state[3, -1] = float(delay_ms) / M_IN_K / BUFFER_NORM_FACTOR

            # 5) 下一段 chunk 在各个码率下的大小（MB）
            next_sizes = np.zeros(BITRATE_LEVELS, dtype=np.float32)
            next_index = min(chunk_idx + 1, total_chunks - 1)
            for br in range(BITRATE_LEVELS):
                next_sizes[br] = video_size[br][next_index]
            state[4, :BITRATE_LEVELS] = next_sizes / M_IN_K / M_IN_K

            # 6) 剩余 chunk 数（归一化）
            remain_chunks = max(total_chunks - (chunk_idx + 1), 0)
            state[5, -1] = min(remain_chunks, CHUNK_TIL_VIDEO_END_CAP) / float(
                CHUNK_TIL_VIDEO_END_CAP
            )

            end_of_video = (chunk_idx == total_chunks - 1)

            # 与 generate_exp_pool.py 保持一致：跳过第一个 step（类似 Pensieve 的做法）
            if chunk_idx > 0:
                exp_pool.add(
                    state=state.copy(),
                    action=action_idx,
                    reward=reward,
                    done=end_of_video,
                )

            last_bit_rate_idx = action_idx

        if len(step_rewards) > 1:
            mean_qoe_excl_first = float(np.mean(step_rewards[1:]))
        else:
            mean_qoe_excl_first = float(np.mean(step_rewards))
        print(
            f'  Trace QoE (mean per chunk, excluding 1st): {mean_qoe_excl_first:.4f}'
        )

    # 保存经验池
    with open(output_path, 'wb') as f:
        pickle.dump(exp_pool, f)

    print(f'Oracle ExperiencePool saved to: {output_path}')
    return exp_pool

# ================= 计算经验池平均 QoE =================

def compute_exp_pool_avg_qoe(exp_pool_path: str) -> float:
    """
    计算已有经验池（exp_pool.pkl）中轨迹的平均 QoE（即平均 reward）。

    说明：
        - 经验池结构为 `plm_special.data.exp_pool.ExperiencePool`：
            - exp_pool.rewards: List[float]，每一步的 QoE = quality - rebuf_penalty - smoothness_penalty
            - exp_pool.dones:   List[bool]，标记 episode 结束位置
        - `generate_exp_pool.py` 在构造经验时已经跳过了每个 episode 的第一个 step，
          因此这里直接对 `rewards` 求均值即可视为“平均每个有效 chunk 的 QoE”。

    Args:
        exp_pool_path: 经验池文件路径，例如 'artifacts/exp_pools/exp_pool.pkl'

    Returns:
        mean_qoe: float，经验池中所有 step 的平均 QoE（reward 的均值）
    """
    if not os.path.exists(exp_pool_path):
        raise FileNotFoundError(f'ExperiencePool file not found: {exp_pool_path}')

    with open(exp_pool_path, 'rb') as f:
        exp_pool = pickle.load(f)

    if not hasattr(exp_pool, 'rewards'):
        raise AttributeError('Loaded object has no attribute "rewards"; is this an ExperiencePool?')

    rewards = np.asarray(exp_pool.rewards, dtype=float)
    if rewards.size == 0:
        print(f'ExperiencePool at {exp_pool_path} is empty. Mean QoE is undefined, return 0.0.')
        return 0.0

    mean_qoe = float(rewards.mean())

    # 统计 episode 数量（以 dones=True 计数）
    dones = getattr(exp_pool, 'dones', None)
    if isinstance(dones, list) and len(dones) == len(rewards):
        episodes = int(sum(1 for d in dones if d))
    else:
        episodes = None

    print(f'Loaded ExperiencePool from: {exp_pool_path}')
    print(f'  Total transitions: {rewards.size}')
    if episodes is not None:
        print(f'  Episodes (by dones): {episodes}')
    print(f'  Average QoE per step (mean reward): {mean_qoe:.6f}')

    return mean_qoe


def merge_exp_pools(input_paths, output_path):
    """
    合并多个 ExperiencePool 为一个新的 ExperiencePool，并保存到 output_path。

    规则：
        - 简单拼接四个列表：states / actions / rewards / dones
        - 不修改任何 reward 或 done 标记

    Args:
        input_paths: List[str]，多个 exp_pool.pkl 的路径
        output_path: str，合并后保存的路径

    Returns:
        merged_pool: 合并后的 ExperiencePool 实例
    """
    from plm_special.data.exp_pool import ExperiencePool

    if not isinstance(input_paths, (list, tuple)) or len(input_paths) == 0:
        raise ValueError('input_paths must be a non-empty list of file paths.')

    merged_pool = ExperiencePool()
    total_transitions = 0

    for p in input_paths:
        if not os.path.exists(p):
            print(f'[merge_exp_pools] Skip non-existent file: {p}')
            continue

        with open(p, 'rb') as f:
            pool = pickle.load(f)

        if not hasattr(pool, 'states') or not hasattr(pool, 'actions') \
           or not hasattr(pool, 'rewards') or not hasattr(pool, 'dones'):
            print(f'[merge_exp_pools] File is not a valid ExperiencePool: {p}')
            continue

        n = len(pool)
        print(f'[merge_exp_pools] Merging {n} transitions from: {p}')

        merged_pool.states.extend(pool.states)
        merged_pool.actions.extend(pool.actions)
        merged_pool.rewards.extend(pool.rewards)
        merged_pool.dones.extend(pool.dones)

        total_transitions += n

    with open(output_path, 'wb') as f:
        pickle.dump(merged_pool, f)

    print(f'[merge_exp_pools] Done. Total merged transitions: {total_transitions}')
    print(f'[merge_exp_pools] Merged ExperiencePool saved to: {output_path}')

    return merged_pool


# ================= 运行入口 =================

if __name__ == '__main__':
    # ================= 测试DP方法 =================
    # 路径
    # trace_path = 'data/traces/train/fcc-train/bus.ljansbakken-oslo-report.2010-09-28_1407CEST.log'
    # video_path = 'data/videos/video1_sizes'

    # if os.path.exists(trace_path) and os.path.exists(video_path):
    #     # 1. 加载数据
    #     print("Loading data...")
    #     times, bws = load_single_trace(trace_path)
    #     video_size = load_video_sizes(video_path)
    #     print("len:", len(times))
    #     for t, bw in zip(times, bws):
    #         print(t, bw)
    #     for r in range(len(video_size)):
    #         print(f"bitrate {r}:")
    #         for i in range(len(video_size[r])):
    #             print(f"  chunk {i}: {video_size[r][i]}")
    #         print()
    #     # 2. 运行 DP
    #     print("\n>>> Running DP Optimizer...")
    #     best_state = solve_optimal_abr(times, bws, video_size)

    #     # 3. 输出结果
    #     if best_state:
    #         print("\n" + "=" * 30)
    #         print("       OPTIMAL RESULT       ")
    #         print("=" * 30)
    #         print(f"Total Reward  : {best_state.reward:.4f}")
    #         print(f"Total Rebuffer: {best_state.total_rebuf:.4f} s")
    #         print(f"Actions (len={len(best_state.history)}):")
    #         print(best_state.history)

    #     else:
    #         print("Error: No solution found.")
    # else:
    #     print(f"Files not found.\nCheck: {trace_path}\nCheck: {video_path}")

    # ================= 生成 Oracle 经验池 =================
    # exp_pool = generate_oracle_exp_pool(
    #     trace_folder='data/traces/test/fcc-test',
    #     #trace_folder='data/traces/train/fcc-train',
    #     video_size_dir='data/videos/video2_sizes',
    #     output_path='artifacts/exp_pools/testv2_exp_pool.pkl',
    #     trace_limit=-1,     
    #     seed=100003,
    # )

    # ================= 合并经验池 =================    
    # merged_exp_pool = merge_exp_pools(
    #     [
    #         'artifacts/exp_pools/oraclev1_exp_pool.pkl',
    #         'artifacts/exp_pools/oraclev2_exp_pool.pkl',
    #     ],
    #     'artifacts/exp_pools/exp_pool.pkl',
    # )

    # ================= 计算 Oracle 经验池的平均 QoE =================
    mean_qoe = compute_exp_pool_avg_qoe('artifacts/exp_pools/testv1_exp_pool.pkl')
    print('Mean QoE:', mean_qoe)