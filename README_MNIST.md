# MNIST DDPM 训练指南 (复现与分析)

本指南将帮助你在 RTX 4090 上完成从环境配置、训练启动到结果分析的完整流程。

## 1. 完整 Pipeline 复现 (Pipeline Reproducibility)

为了保证实验可复现，请严格按照以下步骤操作。我们已经准备好了一键启动脚本。

### 第一步：环境准备
确保你已经按照之前的对话配置好了 conda 环境和 pytorch。
```powershell
conda activate ddpm
# 确保安装了 pytorch 2.x (cuda 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install einops wandb joblib matplotlib tqdm
```

### 第二步：启动训练
使用根目录下的 `run_training.ps1` 脚本（Windows PowerShell）。

**方式 A：使用 WandB (强烈推荐 - 实时监控)**
如果你有 WandB 账号，这是最佳方案。它会自动记录超参数、代码版本、Loss 曲线和生成图片。
```powershell
# 先登录 (仅需一次)
wandb login

# 启动训练
python scripts/train_mnist.py --iterations 30000 --batch_size 128 --log_rate 1000 --checkpoint_rate 5000 --run_name "run_wandb_01" --log_to_wandb True --project_name "ddpm-mnist-repro"
```

**方式 B：本地运行 (无需账号 - 本地文件)**
所有日志和图片保存在本地 `ddpm_logs` 文件夹。
```powershell
python scripts/train_mnist.py --iterations 30000 --batch_size 128 --log_rate 1000 --checkpoint_rate 5000 --run_name "run_local_01" --log_to_wandb False
```

---

## 2. 结果质量与展示 (Quality & Visualization)

### 实时查看 (WandB)
- 打开 WandB 项目页面。
- 点击 "Media" 面板，你可以看到随着训练进行，原本全是噪点的图逐渐变成清晰的数字。
- 这是评估**结果质量**最直观的方法。

### 训练后查看 (本地)
训练脚本会自动在 `ddpm_logs/results/<run_name>/` 目录下保存生成的图片网格。
- **iteration-1000.png**: 应该是一团噪声。
- **iteration-10000.png**: 应该能看到模糊的数字轮廓。
- **iteration-30000.png**: 应该是清晰锐利的手写数字。

你可以运行我为你写的新脚本来批量查看演变过程：
```powershell
python scripts/analyze_results_local.py
```
*(注意：需要修改脚本内的 `run_name` 为你实际的名字)*

---

## 3. 训练过程分析 (Analysis)

### Loss 曲线分析
- **正常曲线**：Loss 应该在最初几千轮迅速下降，然后逐渐变缓并震荡。
- **过拟合 (Overfitting)**：对于生成模型，过拟合通常表现为 Loss 继续下降，但生成的图片开始“死记硬背”训练集（变得千篇一律），或者在验证集上的 Loss 开始上升。
- **欠拟合 (Underfitting)**：生成的图片始终模糊不清，Loss 居高不下。

### 如何判断？
1.  **看 Train Loss vs Test Loss**:
    - 脚本每 1000 轮会计算一次 Test Loss。
    - 如果 WandB 图表中，`train_loss` 在降，但 `test_loss` 开始升高，说明过拟合了，应该停止训练。
    - 对于 MNIST，通常 2~3 万轮左右就能达到很好的平衡。

2.  **看生成多样性**:
    - 观察生成的数字是不是总是“同一个 8”或“同一个 3”。
    - 如果生成的数字看起来样式丰富（不同的笔迹、粗细、倾斜度），说明模型学得很好。

---

## 总结 Check List
- [ ] 运行 `python scripts/train_mnist.py` 成功开始跑数。
- [ ] 看到终端打印 `Iteration 1000: train_loss=...`。
- [ ] 检查 `ddpm_logs/results/` 文件夹，确认生成了 PNG 图片。
- [ ] (可选) 在 WandB 网页上看到 Loss 曲线下降。

