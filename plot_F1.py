import re
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

# 尝试设置中文字体，如果系统没有中文字体则使用英文
try:
    # 尝试查找可用的中文字体
    chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 
                     'Source Han Sans CN', 'Droid Sans Fallback', 'SimHei', 'Microsoft YaHei']
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    
    # 找到第一个可用的中文字体
    font_found = False
    for font in chinese_fonts:
        if font in available_fonts:
            matplotlib.rcParams['font.sans-serif'] = [font] + matplotlib.rcParams['font.sans-serif']
            font_found = True
            break
    
    if not font_found:
        # 如果没有找到中文字体，使用默认字体（英文）
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        print("警告: 未找到中文字体，将使用英文标签")
        USE_CHINESE = False
    else:
        USE_CHINESE = True
except Exception as e:
    print(f"字体设置警告: {e}，将使用默认字体")
    USE_CHINESE = False

matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def extract_f1_scores(log_file):
    """从日志文件中提取所有F1分数"""
    f1_scores = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('Test F1 Score:'):
                # 提取F1分数
                match = re.search(r'Test F1 Score:\s*([\d.]+)', line)
                if match:
                    f1_score = float(match.group(1))
                    f1_scores.append(f1_score)
    
    return f1_scores

def plot_f1_curve(f1_scores, output_file='f1_curve.png'):
    """绘制F1分数变化曲线"""
    if not f1_scores:
        print("未找到F1分数数据！" if USE_CHINESE else "No F1 score data found!")
        return
    
    # 创建x轴（测试序号）
    x = list(range(1, len(f1_scores) + 1))
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    plt.plot(x, f1_scores, marker='o', linestyle='-', linewidth=2, markersize=4)
    
    # 根据字体支持情况选择标签语言
    if USE_CHINESE:
        plt.xlabel('测试序号', fontsize=12)
        plt.ylabel('F1分数', fontsize=12)
        plt.title('测试F1分数变化曲线', fontsize=14, fontweight='bold')
    else:
        plt.xlabel('Test Number', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.title('Test F1 Score Curve', fontsize=14, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    if USE_CHINESE:
        print(f"图表已保存为: {output_file}")
        print(f"共找到 {len(f1_scores)} 个F1分数数据点")
        print(f"最高F1分数: {max(f1_scores):.4f} (第 {f1_scores.index(max(f1_scores)) + 1} 次测试)")
        print(f"最低F1分数: {min(f1_scores):.4f} (第 {f1_scores.index(min(f1_scores)) + 1} 次测试)")
    else:
        print(f"Chart saved as: {output_file}")
        print(f"Found {len(f1_scores)} F1 score data points")
        print(f"Highest F1 score: {max(f1_scores):.4f} (Test #{f1_scores.index(max(f1_scores)) + 1})")
        print(f"Lowest F1 score: {min(f1_scores):.4f} (Test #{f1_scores.index(min(f1_scores)) + 1})")
    
    # 显示图表
    plt.show()

if __name__ == '__main__':
    log_file = 'console.log'
    f1_scores = extract_f1_scores(log_file)
    plot_f1_curve(f1_scores)

