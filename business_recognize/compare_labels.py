#!/usr/bin/env python3
"""
检查GR.csv和result.csv中的标签匹配程度
比较两文件中对应行的label列，输出匹配百分比
"""

import pandas as pd
import sys
import argparse

def compare_labels(gr_file='data/GR.csv', result_file='data/result.csv', match_by_key=True):
    """
    比较GR.csv和result.csv中的标签匹配程度
    
    Args:
        gr_file: GR.csv文件路径
        result_file: result.csv文件路径
        match_by_key: 是否通过前7列匹配行（True），还是按行号对应（False）
    """
    print(f"读取 {gr_file}...")
    gr_df = pd.read_csv(gr_file)
    print(f"GR.csv 共有 {len(gr_df)} 行数据")
    
    print(f"读取 {result_file}...")
    result_df = pd.read_csv(result_file)
    print(f"result.csv 共有 {len(result_df)} 行数据")
    
    # 检查列是否存在
    if 'label' not in gr_df.columns:
        print(f"错误: {gr_file} 中缺少 'label' 列")
        sys.exit(1)
    if 'label' not in result_df.columns:
        print(f"错误: {result_file} 中缺少 'label' 列")
        sys.exit(1)
    
    # 定义前7列作为匹配键
    key_columns = ['protocol', 'hex_src_ip', 'hex_dst_ip', 'src_port', 'dst_port', 'host', 'payload']
    
    if match_by_key:
        # 通过前7列匹配行
        print("\n使用前7列作为匹配键进行匹配...")
        
        # 检查key列是否存在
        for col in key_columns:
            if col not in gr_df.columns or col not in result_df.columns:
                print(f"错误: 缺少匹配列 '{col}'")
                sys.exit(1)
        
        # 构建result_df的查找字典
        result_dict = {}
        for idx, row in result_df.iterrows():
            key = tuple(str(row[col]) for col in key_columns)
            if key in result_dict:
                print(f"警告: result.csv 中发现重复的key (行 {idx+2})")
            result_dict[key] = str(row['label']) if pd.notna(row['label']) else ''
        
        # 匹配并比较
        matched_count = 0
        total_count = 0
        unmatched_keys = []
        
        for idx, row in gr_df.iterrows():
            key = tuple(str(row[col]) for col in key_columns)
            gr_label = str(row['label']) if pd.notna(row['label']) else ''
            
            if key in result_dict:
                result_label = result_dict[key]
                total_count += 1
                if gr_label == result_label:
                    matched_count += 1
                else:
                    if len(unmatched_keys) < 10:  # 只记录前10个不匹配的
                        unmatched_keys.append({
                            'row': idx + 2,
                            'gr_label': gr_label,
                            'result_label': result_label,
                            'host': row.get('host', 'N/A')
                        })
            else:
                print(f"警告: GR.csv 第 {idx+2} 行在 result.csv 中未找到匹配")
        
        print(f"\n匹配统计:")
        print(f"  成功匹配的行数: {total_count}")
        print(f"  未在result.csv中找到匹配的行数: {len(gr_df) - total_count}")
        
    else:
        # 按行号对应比较（假设行数相同且顺序一致）
        print("\n按行号对应进行比较...")
        
        if len(gr_df) != len(result_df):
            print(f"警告: 两个文件行数不同 (GR.csv: {len(gr_df)}, result.csv: {len(result_df)})")
            min_len = min(len(gr_df), len(result_df))
            print(f"将只比较前 {min_len} 行")
            gr_df = gr_df.iloc[:min_len]
            result_df = result_df.iloc[:min_len]
        
        total_count = len(gr_df)
        matched_count = 0
        unmatched_keys = []
        
        for idx in range(total_count):
            gr_label = str(gr_df.iloc[idx]['label']) if pd.notna(gr_df.iloc[idx]['label']) else ''
            result_label = str(result_df.iloc[idx]['label']) if pd.notna(result_df.iloc[idx]['label']) else ''
            
            if gr_label == result_label:
                matched_count += 1
            else:
                if len(unmatched_keys) < 10:  # 只记录前10个不匹配的
                    unmatched_keys.append({
                        'row': idx + 2,
                        'gr_label': gr_label,
                        'result_label': result_label,
                        'host': gr_df.iloc[idx].get('host', 'N/A')
                    })
    
    # 计算匹配率
    if total_count > 0:
        match_rate = (matched_count / total_count) * 100
    else:
        match_rate = 0.0
        print("错误: 没有可比较的行")
        sys.exit(1)
    
    # 输出结果
    print(f"\n{'='*60}")
    print(f"标签匹配结果:")
    print(f"{'='*60}")
    print(f"  总比较行数: {total_count}")
    print(f"  匹配成功: {matched_count} 行")
    print(f"  不匹配: {total_count - matched_count} 行")
    print(f"  匹配率: {match_rate:.2f}%")
    print(f"{'='*60}")
    
    # 显示前几个不匹配的示例
    if unmatched_keys:
        print(f"\n前 {min(10, len(unmatched_keys))} 个不匹配的示例:")
        for i, item in enumerate(unmatched_keys[:10], 1):
            print(f"  {i}. 行 {item['row']} (host: {item['host']}):")
            print(f"     GR.csv label:      {item['gr_label']}")
            print(f"     result.csv label:   {item['result_label']}")
    
    return match_rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='比较GR.csv和result.csv中的标签匹配程度')
    parser.add_argument('--gr-file', type=str, default='data/GR.csv', help='GR.csv文件路径')
    parser.add_argument('--result-file', type=str, default='data/result.csv', help='result.csv文件路径')
    parser.add_argument('--by-row', action='store_true', help='按行号对应比较（默认：通过前7列匹配）')
    
    args = parser.parse_args()
    
    match_by_key = not args.by_row
    match_rate = compare_labels(args.gr_file, args.result_file, match_by_key)
    
    # 退出码：0表示完全匹配，1表示有差异
    sys.exit(0 if match_rate == 100.0 else 1)

