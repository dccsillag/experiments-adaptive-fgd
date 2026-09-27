import aim
import torch

from src.functional_radiance.models.nerf_pytorch.train import train

if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    aim_run = aim.Run(experiment="nerf")

    args_overrides = {}
    args_overrides["render_only"] = True

    args_overrides["render_fold"] = "train"
    train(args_overrides, aim_run)

    args_overrides["render_fold"] = "test"
    train(args_overrides, aim_run)

    args_overrides["render_fold"] = "val"
    train(args_overrides, aim_run)
