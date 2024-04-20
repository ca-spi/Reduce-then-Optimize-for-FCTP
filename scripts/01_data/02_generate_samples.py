""" Solve FCTP instances to generate samples. """

import argparse
from datetime import timedelta
import gzip
import os
import pickle as pkl
import random
import re
from time import time

import numpy as np

from core.data_processing.data_utils import load_instance
from core.data_processing.sample_generation import generate_sample


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instance_dir",
        help="Path to instance directory.",
        type=str,
        default=os.path.join("data", "instances"),
    )
    parser.add_argument(
        "--grb_timeout", help="Gurobi runtime limit.", type=int, default=600
    )
    parser.add_argument("--seed", help="Random seed number.", type=int, default=0)
    parser.add_argument(
        "--num_threads",
        help="Control number of threads used by Gurobi.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--keep_subopt",
        default=False,
        action="store_true",
        help="Indicator whether also solutions without proven optimality shall be saved.",
    )
    parser.add_argument(
        "--sample_dir",
        help="Path to directory to save samples",
        type=str,
        default=os.path.join("data", "samples"),
    )
    parser.add_argument(
        "--reset",
        default=False,
        action="store_true",
        help="Indicator whether old samples shall be overwritten by new samples.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Indicator whether solver output should be written to console.",
    )
    args = parser.parse_args()
    return args


def process_instance(
    instance_path,
    grb_timeout,
    keep_subopt,
    sample_path,
    seed=0,
    grb_threads=None,
    verbosity=False,
):
    """Wrapper function read and solve instance and save results.

    Parameters
    ----------
    instance_path: str
        Path to instance file.
    grb_timeout: int
        Gurobi runtime limit
    keep_subopt: bool
        Indicator whether also solutions without proven optimality shall be saved.
    sample_path: str
        Path to sample file to save results.
    seed: int, optional
        Random seed number. Default is 0.
    grb_threads: int, optional
        Control number of threads used by Gurobi.
    verbosity: bool, optional
        Indicator whether solver output should be written to console.

    Returns
    -------
    None

    """
    # Load instance
    instance = load_instance(instance_path)

    # Solve to optimality
    sample = generate_sample(
        instance=instance,
        grb_timeout=grb_timeout,
        grb_threads=grb_threads,
        grb_seed=seed,
        verbosity=verbosity,
    )
    if sample["opt_status"] != 2 and not keep_subopt:
        return

    # Save sample
    sample["instance_path"] = instance_path
    sample["instance"] = sample["instance"].to_dict()
    with gzip.open(sample_path, "wb") as file:
        pkl.dump(
            sample,
            file,
        )


def main():
    args = parse_args()
    print(args)

    # Set all random seeds
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    # Set up directory
    os.makedirs(args.sample_dir, exist_ok=True)

    # Delete existing samples if desired
    prev_samples = os.listdir(args.sample_dir)
    if args.reset and len(prev_samples) > 0:
        for f in prev_samples:
            os.remove(os.path.join(args.sample_dir, f))

    # Solve instances to optimality and save solution
    instance_paths = [
        os.path.join(args.instance_dir, filename)
        for filename in os.listdir(args.instance_dir)
    ]
    for instance_path in instance_paths:
        instance_id = int(re.findall("[0-9]+", instance_path.split("/")[-1])[0])
        sample_filename = f"sample_{instance_id}.pkl.gz"
        sample_path = os.path.join(args.sample_dir, sample_filename)

        # Skip instance if sample already exists and should not be overwritten
        if not args.reset and sample_filename in prev_samples:
            print(f"Sample for instance {instance_id} already existed -> aborting")
            continue

        # Print progress
        print(f"Processing instance {instance_id}...")
        start_time = time()

        process_instance(
            instance_path=instance_path,
            grb_timeout=args.grb_timeout,
            keep_subopt=args.keep_subopt,
            sample_path=sample_path,
            seed=seed,
            grb_threads=args.num_threads,
            verbosity=args.verbose,
        )

        time_passed = time() - start_time
        print(
            f"...Processing instance {instance_id} finished after {timedelta(seconds=time_passed)} and saved to {sample_path}"
        )


if __name__ == "__main__":
    main()
