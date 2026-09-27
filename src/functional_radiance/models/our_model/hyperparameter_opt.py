import argparse
import getpass
import json
import os
import socket
import subprocess
import tempfile
from pathlib import Path

import optuna
from icecream import ic  # type: ignore

from src.functional_radiance.utils.log import get_readable_time

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "--n_trials",
    type=int,
    required=True,
    help="how many trials to run for the hyperparameter sweep",
)

parser.add_argument(
    "--scene",
    type=str,
    required=True,
    help="which scene to train on",
)
parser.add_argument(
    "--setting",
    type=str,
    choices=["full", "small"],
    required=True,
    help="which size of the scene to use",
)

parser.add_argument(
    "-r",
    "--representation",
    default="proper_voxel_grid",
    choices=["proper_voxel_grid", "proper_octree"],
    help="which representation to use",
)

parser.add_argument(
    "--grid_size",
    default=60,
    type=int,
    help="if voxel_grid representation is used, determines grid size",
)

parser.add_argument(
    "-e",
    "--max_epochs",
    type=int,
    default=50,
)

parser.add_argument(
    "--batch_size_render",
    type=int,
    default=1,
)
parser.add_argument(
    "--batch_size_gradient",
    type=int,
    default=8,
)

parser.add_argument(
    "--min_density",
    type=float,
    help="Minimum density for clamping; if none is passed, don't clamp",
)
parser.add_argument(
    "--max_density",
    type=float,
    help="Maximum density for clamping; if none is passed, don't clamp",
)

args = parser.parse_args()


study_id = get_readable_time()
container_dir = Path(f"models/our_model/sweep/{args.scene}-{args.setting}/{study_id}")
os.makedirs(container_dir)


def run_trial(trial):
    trial_dir = container_dir / f"trial-{get_readable_time()}"

    argv = [
        "python",
        "src/models/our_model/train.py",
        "--output",
        trial_dir,
        "--scene",
        args.scene,
        "--setting",
        args.setting,
        "--representation",
        args.representation,
        "--grid_size",
        args.grid_size,
        "--max_epochs",
        args.max_epochs,
        "--batch_size_render",
        args.batch_size_render,
        "--batch_size_gradient",
        args.batch_size_gradient,
        "--lr",
        trial.suggest_float("lr", 1.0, 50.0),
        "--sh_deg",
        trial.suggest_int("sh_deg", 0, 2),
        "--momentum" if trial.suggest_categorical("momentum", [True, False]) else None,
        "--n_samples_for_antialiasing",
        trial.suggest_int("n_samples_for_antialiasing", 1, 20),
        "--regularization_lp_p",
        trial.suggest_float("regularization_lp_p", 0.0, 2.0),
        "--regularization_lp_lambda",
        trial.suggest_float("regularization_lp_lambda", 0.0, 0.2),
        "--regularization_lpray_p",
        trial.suggest_float("regularization_lpray_p", 0.0, 2.0),
        "--regularization_lpray_lambda",
        trial.suggest_float("regularization_lpray_lambda", 0.0, 0.2),
        *(
            ["--min_density", args.min_density]
            if args.min_density is not None
            else [None]
        ),
        *(
            ["--max_density", args.max_density]
            if args.max_density is not None
            else [None]
        ),
    ]
    argv = [str(arg) for arg in argv if arg is not None]
    subprocess.run(argv, check=True)

    with open(trial_dir / "psnr.json") as file:
        psnr_history = json.load(file)

    return max(psnr_history["val"])


study = optuna.create_study(
    # storage=ic("sqlite://" + str((container_dir / "db.sqlite3").resolve())),
    # study_name=study_id,
    direction="maximize",
)
study.optimize(run_trial, n_trials=args.n_trials)

# print()
# print(f"Run `optuna-dashboard {container_dir / 'db.sqlite3'}` to open the Optuna dashboard")
# print()

trial = study.best_trial
print(f"Best PSNR: {trial.value}")
print(f"Best hyperparameters: {trial.params}")
