import argparse
import torch
import torchvision
from ddpm import script_utils
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint (.pth)")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of images to generate")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for generation")
    parser.add_argument("--output_dir", type=str, default="generated_samples", help="Directory to save images")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    # Add necessary args for model construction
    parser.add_argument("--img_channels", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=128)
    parser.add_argument("--channel_mults", type=str, default="1,2,2,2") # We need to parse this manually or pass tuple
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--time_emb_dim", type=int, default=512) # 128*4
    parser.add_argument("--norm", type=str, default="gn")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="silu")
    parser.add_argument("--attention_resolutions", type=str, default="1")
    
    # Diffusion args
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--schedule_low", type=float, default=1e-4)
    parser.add_argument("--schedule_high", type=float, default=0.02)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_update_rate", type=int, default=1)
    parser.add_argument("--loss_type", type=str, default="l2")
    parser.add_argument("--use_labels", type=bool, default=False)
    parser.add_argument("--num_classes", type=int, default=10)

    args = parser.parse_args()
    
    # Manually fix tuple args that argparse reads as strings if not using the original script_utils
    # But here we can reuse script_utils.diffusion_defaults() if we reconstruct the args object correctly
    # or just manually construct the model.
    # Let's try to reuse script_utils as much as possible to match training.
    
    # Re-construct args to match what script_utils.get_diffusion_from_args expects
    # Note: script_utils expects a Namespace object with specific attributes
    
    # Fix channel_mults and attention_resolutions which are tuples
    # In train_mnist.py/script_utils defaults:
    # channel_mults=(1, 2, 2, 2)
    # attention_resolutions=(1,)
    # We will hardcode these to match the MNIST training defaults unless you changed them
    args.channel_mults = (1, 2, 2, 2)
    args.attention_resolutions = (1,)
    
    device = torch.device(args.device)
    
    print(f"Loading model from {args.model_path}...")
    
    # Load model
    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        diffusion.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model architecture arguments match training configuration.")
        return

    diffusion.eval()
    
    print(f"Generating {args.num_samples} samples...")
    
    with torch.no_grad():
        if args.use_labels:
            # Generate random labels if using class conditioning
            y = torch.randint(0, 10, (args.num_samples,), device=device)
            samples = diffusion.sample(args.num_samples, device, y=y)
        else:
            samples = diffusion.sample(args.num_samples, device)
            
    # Normalize to [0, 1]
    samples = ((samples + 1) / 2).clip(0, 1)
    
    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save grid
    grid_path = os.path.join(args.output_dir, "grid_50.png")
    torchvision.utils.save_image(samples, grid_path, nrow=10)
    print(f"Saved grid image to {grid_path}")
    
    # Save individual images
    for i, sample in enumerate(samples):
        img_path = os.path.join(args.output_dir, f"sample_{i:03d}.png")
        torchvision.utils.save_image(sample, img_path)
    
    print(f"Saved {args.num_samples} individual images to {args.output_dir}")

if __name__ == "__main__":
    main()

