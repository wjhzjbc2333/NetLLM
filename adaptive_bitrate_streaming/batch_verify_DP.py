import os
import glob
import time
import numpy as np

# ================= 引入外部 DP 模块 =================
# 确保 DP.py 在同一目录下，或者在 PYTHONPATH 中
from DP import solve_optimal_abr, load_single_trace, load_video_sizes, State

# ================= 重新声明配置 (必须与 DP1 保持一致) =================
BITS_IN_BYTE = 8.0
B_IN_MB = 1000000.0
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # ms
MILLISECONDS_IN_SECOND = 1000.0
VIDEO_CHUNK_LEN = 4000.0  # ms
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
M_IN_K = 1000.0
REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1.0


def evaluate_and_log(actions, times, bws, video_size, log_file_path):
    """
    执行动作序列，并将每一步的详细指标写入文件 (Tab分隔)
    新增返回: first_chunk_reward 用于统计时剔除
    """
    # 准备文件
    result_file = open(log_file_path, 'w')

    # 初始化状态
    ptr = 1
    last_time = times[0]
    buffer_size_ms = 0.0  # 内部计算用 ms

    total_reward = 0.0
    total_rebuf_sec = 0.0
    prev_bitrate = VIDEO_BIT_RATE[0]  # 假设初始码率

    first_chunk_reward = 0.0  # 用于记录第一步的 reward

    # print(f"Logging results to: {log_file_path}") # 批量处理时注释掉，避免刷屏

    for chunk_idx, action in enumerate(actions):
        # 1. 获取数据
        curr_bitrate = VIDEO_BIT_RATE[action]
        video_chunk_size = video_size[action][chunk_idx]

        # 2. 模拟下载 (Physics)
        downloaded = 0.0
        delay = 0.0
        curr_ptr = ptr
        curr_last_time = last_time

        while True:
            bw = bws[curr_ptr]
            throughput = bw * B_IN_MB / BITS_IN_BYTE
            duration = times[curr_ptr] - curr_last_time

            if duration <= 0:
                curr_ptr += 1
                if curr_ptr >= len(bws): curr_ptr = 1; curr_last_time = 0.0
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
                if curr_ptr >= len(bws): curr_ptr = 1; curr_last_time = 0.0

        # 更新时间指针
        ptr = curr_ptr
        last_time = curr_last_time

        # 3. 计算状态变化
        delay_ms = delay * MILLISECONDS_IN_SECOND + LINK_RTT
        delay_sec = delay_ms / MILLISECONDS_IN_SECOND

        # 计算卡顿 (Rebuffer)
        rebuf_ms = max(delay_ms - buffer_size_ms, 0.0)
        rebuf_sec = rebuf_ms / MILLISECONDS_IN_SECOND

        # 更新缓冲区 (Buffer)
        buffer_size_ms = max(buffer_size_ms - delay_ms, 0.0) + VIDEO_CHUNK_LEN
        buffer_size_sec = buffer_size_ms / MILLISECONDS_IN_SECOND

        # 4. 计算 Reward
        r_quality = curr_bitrate / M_IN_K
        r_rebuf = REBUF_PENALTY * rebuf_sec

        if chunk_idx == 0:
            smoothness_diff = 0.0
        else:
            smoothness_diff = abs(curr_bitrate - prev_bitrate) / M_IN_K

        r_smooth = SMOOTH_PENALTY * smoothness_diff

        reward = r_quality - r_rebuf - r_smooth

        # 记录第一步的 reward
        if chunk_idx == 0:
            first_chunk_reward = reward

        total_reward += reward
        total_rebuf_sec += rebuf_sec

        # 5. 写入日志
        result_file.write(
            str(last_time) + '\t' +  # Time Stamp (Seconds)
            str(curr_bitrate) + '\t' +  # Bitrate (Kbps)
            str(buffer_size_sec) + '\t' +  # Buffer (Seconds)
            str(rebuf_sec) + '\t' +  # Rebuf (Seconds)
            str(video_chunk_size) + '\t' +  # Chunk Size (Bytes)
            str(delay_sec) + '\t' +  # Delay (Seconds)
            str(r_smooth) + '\t' +  # Smoothness Penalty
            str(reward) + '\n'  # Step Reward
        )

        prev_bitrate = curr_bitrate

    result_file.close()
    # 返回值增加 first_chunk_reward
    return total_reward, first_chunk_reward


# ================= 批量处理逻辑 =================
if __name__ == '__main__':
    # 1. 定义路径
    TRACE_FOLDER = 'data/traces/test/fcc-test'
    VIDEO_PATH = 'data/videos/video1_sizes'
    OUTPUT_FOLDER = 'artifacts/results/fcc-test_video1/trace_num_100_fixed_True/DP/result_log'

    # 2. 检查并创建输出目录
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        print(f"Created output directory: {OUTPUT_FOLDER}")

    # 3. 加载视频数据 (所有 Trace 共用)
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video path not found: {VIDEO_PATH}")
        exit()

    print("Loading video sizes...")
    video_size = load_video_sizes(VIDEO_PATH)

    # 4. 获取 Trace 文件列表
    trace_files = glob.glob(os.path.join(TRACE_FOLDER, '*'))
    total_files = len(trace_files)

    if total_files == 0:
        print(f"No trace files found in {TRACE_FOLDER}")
        exit()

    print(f"Found {total_files} trace files. Starting batch processing...")
    start_time_all = time.time()

    # --- 全局统计变量 ---
    grand_total_reward = 0.0
    grand_total_chunks = 0
    valid_trace_count = 0

    # 5. 循环处理每个 Trace
    for i, trace_file in enumerate(trace_files):
        trace_name = os.path.basename(trace_file)
        log_filename = 'log_' + trace_name
        output_path = os.path.join(OUTPUT_FOLDER, log_filename)

        # 打印进度
        print(f"[{i + 1}/{total_files}] Processing: {trace_name} ...", end='', flush=True)

        try:
            # A. 加载 Trace
            times, bws = load_single_trace(trace_file)
            if not times:
                print(" [Skipped: Invalid Trace]")
                continue

            # B. 调用 DP 求解 (Oracle)
            best_state = solve_optimal_abr(times, bws, video_size, buffer_quantization_ms=100.0)

            if best_state:
                actions = best_state.history

                # C. 回放并记录日志 (Replay & Log)
                real_reward, first_reward = evaluate_and_log(actions, times, bws, video_size, output_path)

                # D. 累加全局统计数据 (核心修改：剔除第一步)
                num_chunks = len(actions)

                # 剔除第一步reward
                if num_chunks > 1:
                    adjusted_reward = real_reward - first_reward
                    adjusted_chunks = num_chunks - 1

                    grand_total_reward += adjusted_reward
                    grand_total_chunks += adjusted_chunks

                    valid_trace_count += 1

                    # 打印当前 trace 的平均分 (已剔除第一步)
                    avg_trace_reward = adjusted_reward / adjusted_chunks
                    print(
                        f" Done. Total Reward: {adjusted_reward:.2f}, Avg/Chunk (exc. 1st): {avg_trace_reward:.4f}")
                else:
                    print(f" Done. (Skipped stats: too few chunks {num_chunks})")

            else:
                print(" Failed (No Solution).")

        except Exception as e:
            print(f" Error: {e}")

    # 6. 结束统计
    duration = time.time() - start_time_all

    # 计算全局单步平均 Reward
    avg_reward_per_step = 0.0
    if grand_total_chunks > 0:
        avg_reward_per_step = grand_total_reward / grand_total_chunks

    print("\n" + "=" * 50)
    print("Batch Processing Complete!")
    print(f"Total Time: {duration:.2f} seconds")
    print(f"Total Processed Traces: {valid_trace_count}")
    print("-" * 50)
    print(f"Total Reward (All Traces, excluding 1st chunk): {grand_total_reward:.4f}")
    print(f"Total Chunks Processed (excluding 1st chunk): {grand_total_chunks}")
    print(f"Average Reward per Step (QoE/chunk): {avg_reward_per_step:.4f}")
    print("-" * 50)
    print(f"Results saved to: {OUTPUT_FOLDER}")
    print("=" * 50)