$env:PYTHONPATH="."
# 1. 创建虚拟环境 (如果尚未创建)
# conda create -n ddpm python=3.10 -y
# conda activate ddpm

# 2. 安装依赖 (适配你的服务器环境: PyTorch 2.0 + CUDA 11.8)
# 注意：你的服务器截图显示是 CUDA 11.8，所以 index-url 用 cu118
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install einops wandb joblib matplotlib tqdm

# 3. 运行训练
# 参数说明:
# --iterations 30000: 训练3万轮，对MNIST足够了 (约40分钟)
# --batch_size 128: 4090显存很大，可以适当调大
# --log_rate 1000: 每1000轮输出一次日志和保存图片
# --checkpoint_rate 5000: 每5000轮保存一次模型权重
# --project_name ddpm-mnist-repro: WandB的项目名称
# --run_name run1: 实验名称，方便复现

#若使用wandb则需要wandb login
#export PYTHONPATH=$PYTHONPATH:.
# 方式一：不使用 WandB (本地看图)
python scripts/train_mnist.py --iterations 30000 --batch_size 128 --log_rate 1000 --checkpoint_rate 5000 --run_name "run_local_01" --log_to_wandb False

# 方式二：使用 WandB (推荐，手机实时监控，需先运行 wandb login)
# python scripts/train_mnist.py --iterations 30000 --batch_size 128 --log_rate 1000 --checkpoint_rate 5000 --run_name "run_wandb_01" --log_to_wandb True --project_name "ddpm-mnist-repro"
