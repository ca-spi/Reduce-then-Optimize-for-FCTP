""" Generate FCTP instances. """

import argparse
from functools import partial
import gzip
import os
import pickle as pkl
import random
import re

import numpy as np

from core.data_processing.instance_generation import generate_random_fctp_instance
from core.data_processing.instance_generation import (
    generate_random_capacitated_fctp_instance,
)
from core.data_processing.instance_generation import (
    generate_random_fixedstep_fctp_instance,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_instances",
        help="Number of instances to be generated",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--instance_type",
        help="Type of instance to be generated.",
        type=str,
        choices=["fctp", "c-fctp", "fs-fctp"],
        default="fctp",
    )
    parser.add_argument(
        "--size", help="Number of suppliers/customers in FCTP", type=int, default=15
    )
    parser.add_argument(
        "--max_quantity",
        help="Maximum supply/demand quantity.",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--cost_structure",
        help="Type of cost strcuture",
        type=str,
        choices=["agarwal-aneja", "roberti", "euclidean"],
        default="agarwal-aneja",
    )
    parser.add_argument(
        "--theta",
        help="Proportion of variable and fixed costs",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--supply_demand_ratio",
        help="Proportion of total supply to total demand",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--capacity_range",
        help="Range to sample relative edge capacity from (in %).",
        type=int,
        nargs=2,
        default=[50, 150],
    )
    parser.add_argument(
        "--dir",
        help="Path to directory to save instances",
        type=str,
        default=os.path.join("data", "instances"),
    )
    parser.add_argument(
        "--reset",
        default=False,
        action="store_true",
        help="Indicator whether old instances shall be deleted and overwritten by new instances.",
    )
    parser.add_argument(
        "--one_instance_dirs",
        default=False,
        action="store_true",
        help="Indicator whether each instance should be stored in a separate subdirectory.",
    )
    parser.add_argument("--seed", help="Random seed number", type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)

    # Set all random seeds
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    # Set up directory
    os.makedirs(args.dir, exist_ok=True)

    # Configure instance generation function
    if args.instance_type == "c-fctp":
        sample_fun = partial(
            generate_random_capacitated_fctp_instance,
            capacity_range=args.capacity_range,
        )
    elif args.instance_type == "fs-fctp":
        sample_fun = generate_random_fixedstep_fctp_instance
    else:
        sample_fun = generate_random_fctp_instance

    # Check whether there are already instances in this directory and either delete them or
    # set instance index accordingly
    start_index = 0
    prev_instances = os.listdir(args.dir)
    if len(prev_instances) > 0:
        if args.reset:
            for f in prev_instances:
                os.remove(os.path.join(args.dir, f))
        else:
            start_index = (
                max([int(re.findall("[0-9]+", f)[0]) for f in prev_instances]) + 1
            )
            print(f"Non-empty instance directory, starting at index {start_index}")

    # Generate and save instances
    print(f"Generating {args.num_instances} instances...")
    instance_id = start_index
    for i in range(args.num_instances):
        # Print progress
        if not i % 1000 and i > 0:
            print(f"{i} instances generated...")
        # generate instance
        if args.supply_demand_ratio < 0:
            supply_demand_ratio = np.random.uniform(1.1, 1.3)
        else:
            supply_demand_ratio = args.supply_demand_ratio
        instance = sample_fun(
            size=args.size,
            max_quantity=args.max_quantity,
            cost_structure=args.cost_structure,
            theta=args.theta,
            balance=True,
            supply_demand_ratio=supply_demand_ratio,
        )
        # save instance
        if args.one_instance_dirs:
            instance_dir = os.path.join(args.dir, f"instance_{instance_id}")
            os.makedirs(instance_dir, exist_ok=True)
            path = os.path.join(instance_dir, f"instance_{instance_id}.pkl.gz")
        else:
            path = os.path.join(args.dir, f"instance_{instance_id}.pkl.gz")
        with gzip.open(path, "wb") as file:
            pkl.dump(instance.to_dict(), file)
        instance_id += 1

    print(f"Finished. {i+1} instances generated.")


if __name__ == "__main__":
    main()
