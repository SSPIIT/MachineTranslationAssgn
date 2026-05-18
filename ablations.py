"""
ablations.py - Runner for all W&B report experiments (Section 2)
DA6401 Assignment 3

Runs all five ablation experiments and logs results to W&B.
Usage: python ablations.py --experiment <1|2|3|4|5|all>
"""

import os
import sys
import argparse
import subprocess


BASE_CMD = "python train.py --epochs 15 --batch_size 128"


EXPERIMENTS = {
    # 2.1 Noam vs Fixed LR
    "2_1_noam": f"{BASE_CMD} --run_name exp_noam --label_smoothing 0.1",
    "2_1_fixed_lr": f"{BASE_CMD} --run_name exp_fixed_lr --fixed_lr 1e-4 --label_smoothing 0.1",

    # 2.2 Scaling factor ablation handled inside train.py via model variant
    # We use a special flag --no_scale_attn passed to model
    "2_2_with_scale":    f"{BASE_CMD} --run_name exp_scale_attn --label_smoothing 0.1",
    "2_2_without_scale": f"{BASE_CMD} --run_name exp_no_scale_attn --no_scale_attn --label_smoothing 0.1",

    # 2.4 Sinusoidal vs Learned PE
    "2_4_sinusoidal": f"{BASE_CMD} --run_name exp_sinusoidal_pe --label_smoothing 0.1",
    "2_4_learned_pe": f"{BASE_CMD} --run_name exp_learned_pe --learned_pe --label_smoothing 0.1",

    # 2.5 Label smoothing vs no smoothing
    "2_5_smoothing":    f"{BASE_CMD} --run_name exp_label_smooth --label_smoothing 0.1",
    "2_5_no_smoothing": f"{BASE_CMD} --run_name exp_no_label_smooth --label_smoothing 0.0",
}


def run(cmd):
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print('='*60)
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"[WARNING] Command exited with code {ret.returncode}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='all',
                        choices=list(EXPERIMENTS.keys()) + ['all'],
                        help='Which experiment to run')
    args = parser.parse_args()

    if args.experiment == 'all':
        for name, cmd in EXPERIMENTS.items():
            run(cmd)
    else:
        run(EXPERIMENTS[args.experiment])