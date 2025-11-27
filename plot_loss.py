import matplotlib.pyplot as plt
import numpy as np

def plot_loss(loss_file='loss.txt', save_path=None, show_plot=True):
    """
    读取损失文件并绘制损失曲线
    
    Args:
        loss_file: 损失文件路径，默认为 'loss.txt'
        save_path: 保存图片的路径，如果为None则不保存
        show_plot: 是否显示图片，默认为True
    """
    # 读取损失数据
    try:
        losses = np.loadtxt(loss_file)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {loss_file}")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 创建步数数组（从1开始）
    steps = np.arange(1, len(losses) + 1)
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 绘制损失曲线
    plt.plot(steps, losses, linewidth=0.5, alpha=0.7, color='blue', label='Training Loss')
    
    # 如果数据点太多，可以绘制移动平均线
    if len(losses) > 100:
        window_size = min(100, len(losses) // 10)
        moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        moving_steps = steps[window_size-1:]
        plt.plot(moving_steps, moving_avg, linewidth=2, color='red', 
                label=f'Moving Average (window={window_size})')
    
    # 设置标签和标题
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # 设置坐标轴格式
    plt.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    # 显示图片
    if show_plot:
        plt.show()
    
    # 打印统计信息
    print(f"\n损失统计信息:")
    print(f"  总步数: {len(losses):,}")
    print(f"  初始损失: {losses[0]:.6f}")
    print(f"  最终损失: {losses[-1]:.6f}")
    print(f"  最小损失: {np.min(losses):.6f} (步数: {np.argmin(losses) + 1})")
    print(f"  最大损失: {np.max(losses):.6f} (步数: {np.argmax(losses) + 1})")
    print(f"  平均损失: {np.mean(losses):.6f}")
    print(f"  标准差: {np.std(losses):.6f}")

if __name__ == "__main__":
    # 可以在这里修改参数
    plot_loss(
        loss_file='loss.txt',
        save_path='loss_curve.png',  # 保存为 loss_curve.png，设为 None 则不保存
        show_plot=True
    )

