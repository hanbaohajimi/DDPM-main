import matplotlib.pyplot as plt
import re
import os
import glob
from PIL import Image
import math

def parse_log_file(log_path):
    """从训练日志中提取 loss 数据 (如果不使用 WandB)"""
    train_losses = []
    test_losses = []
    iterations = []
    
    # 简单的模拟：实际情况中你可能需要把打印到终端的日志重定向到文件
    # 或者直接读取 wandb 的本地数据
    # 这里我们演示如何从保存的图片文件名推断进度，或者假设你有保存的 log.txt
    pass 

def visualize_results(run_name, log_dir="ddpm_logs"):
    """
    可视化本地保存的训练结果
    1. 展示生成的样本随时间的变化
    2. 如果有保存 loss 记录，绘制曲线
    """
    results_dir = os.path.join(log_dir, "results", run_name)
    if not os.path.exists(results_dir):
        print(f"找不到结果目录: {results_dir}")
        return

    # 1. 获取所有生成的图片
    images = sorted(glob.glob(os.path.join(results_dir, "iteration-*.png")), key=os.path.getmtime)
    
    if not images:
        print("没有找到生成的图片。")
        return

    print(f"找到 {len(images)} 张生成样本图片。")

    # 选择展示部分图片（例如开始、中间、结束）
    num_show = min(5, len(images))
    indices = [int(i) for i in list(map(lambda x: x * (len(images)-1) / (num_show-1), range(num_show)))]
    # 去重并排序
    indices = sorted(list(set(indices)))
    
    plt.figure(figsize=(20, 4))
    for i, idx in enumerate(indices):
        img_path = images[idx]
        iter_num = re.search(r"iteration-(\d+)", img_path).group(1)
        
        img = Image.open(img_path)
        plt.subplot(1, len(indices), i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Iter {iter_num}")
        plt.axis('off')
    
    plt.suptitle(f"Generation Quality Evolution ({run_name})")
    plt.tight_layout()
    plt.show()
    print("已显示生成样本演变图。")

if __name__ == "__main__":
    # 使用示例
    # 请将 "run_local_01" 替换为你实际设置的 run_name
    run_name = "run_local_01" 
    visualize_results(run_name)
    
    print("\n提示: 如果你使用了 WandB，请直接在网页端查看 Loss 曲线和高清大图，体验更好。")

