import argparse
import datetime
import torch
import wandb
import torchvision

from torch.utils.data import DataLoader
from torchvision import datasets
from ddpm import script_utils


def main():
    args = create_argparser().parse_args()
    device = args.device

    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)

        if args.model_checkpoint is not None:
            diffusion.load_state_dict(torch.load(args.model_checkpoint))
        if args.optim_checkpoint is not None:
            optimizer.load_state_dict(torch.load(args.optim_checkpoint))

        if args.log_to_wandb:
            if args.project_name is None:
                raise ValueError("args.log_to_wandb set to True but args.project_name is None")

            run = wandb.init(
                project=args.project_name,
                entity='1841672784-hdu',
                config=vars(args),
                name=args.run_name,
            )
            wandb.watch(diffusion)

        batch_size = args.batch_size

        # MNIST is 28x28, so we pad to 32x32
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Pad(2),
            script_utils.get_transform(),
        ])

        train_dataset = datasets.MNIST(
            root='./mnist_train',
            train=True,
            download=True,
            transform=transform,
        )

        test_dataset = datasets.MNIST(
            root='./mnist_test',
            train=False,
            download=True,
            transform=transform,
        )

        train_loader = script_utils.cycle(DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2,
        ))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, num_workers=2)
        
        acc_train_loss = 0

        for iteration in range(1, args.iterations + 1):
            diffusion.train()

            x, y = next(train_loader)
            x = x.to(device)
            y = y.to(device)

            if args.use_labels:
                loss = diffusion(x, y)
            else:
                loss = diffusion(x)

            acc_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            diffusion.update_ema()
            
            if iteration % args.log_rate == 0:
                test_loss = 0
                with torch.no_grad():
                    diffusion.eval()
                    for x, y in test_loader:
                        x = x.to(device)
                        y = y.to(device)

                        if args.use_labels:
                            loss = diffusion(x, y)
                        else:
                            loss = diffusion(x)

                        test_loss += loss.item()
                
                if args.use_labels:
                    samples = diffusion.sample(10, device, y=torch.arange(10, device=device))
                else:
                    samples = diffusion.sample(10, device)
                
                # Normalize samples to [0, 1] for saving/logging
                samples_norm = ((samples + 1) / 2).clip(0, 1)

                test_loss /= len(test_loader)
                acc_train_loss /= args.log_rate

                if args.log_to_wandb:
                    # WandB expects numpy arrays (H, W, C)
                    samples_wandb = samples_norm.permute(0, 2, 3, 1).cpu().numpy()
                    wandb.log({
                        "test_loss": test_loss,
                        "train_loss": acc_train_loss,
                        "samples": [wandb.Image(sample) for sample in samples_wandb],
                    })
                else:
                    # Save local images
                    import os
                    results_dir = f"{args.log_dir}/results/{args.run_name}"
                    os.makedirs(results_dir, exist_ok=True)
                    torchvision.utils.save_image(samples_norm, f"{results_dir}/iteration-{iteration}.png", nrow=5)
                    print(f"Iteration {iteration}: train_loss={acc_train_loss:.4f}, test_loss={test_loss:.4f}. Saved sample to {results_dir}")

                acc_train_loss = 0
            
            if iteration % args.checkpoint_rate == 0:
                model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-model.pth"
                optim_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-optim.pth"

                # Ensure log directory exists
                import os
                os.makedirs(args.log_dir, exist_ok=True)

                torch.save(diffusion.state_dict(), model_filename)
                torch.save(optimizer.state_dict(), optim_filename)
        
        if args.log_to_wandb:
            run.finish()
    except KeyboardInterrupt:
        if args.log_to_wandb:
            run.finish()
        print("Keyboard interrupt, run finished early")


def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")
    defaults = dict(
        learning_rate=2e-4,
        batch_size=128,
        iterations=800000,

        log_to_wandb=False, # Default to False for easier local run
        log_rate=1000,
        checkpoint_rate=1000,
        log_dir="ddpm_logs", # Local directory
        project_name="ddpm-mnist",
        run_name=run_name,

        model_checkpoint=None,
        optim_checkpoint=None,

        schedule_low=1e-4,
        schedule_high=0.02,

        device=device,
    )
    defaults.update(script_utils.diffusion_defaults())
    
    # Overwrite defaults for MNIST
    defaults['img_channels'] = 1
    defaults['img_size'] = 32 # Padded from 28

    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

