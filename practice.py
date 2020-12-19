import os
import random
import argparse
import time

import numpy as np
# TODO: (1) import wandb

# TODO: update constants if needed
from constants import PROJECT

# DON'T LOOK ME!!! IT'S A KIND OF CHEAT!!!
from model import DummyModel


def main(args):
    # fix seed
    if args.seed is not None:
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # TODO: (1) init wandb run
    # - (1.1) specify "your project"
    # - (1.2) upload experiment setting, such as `args`
    # ref: https://docs.wandb.com/library/init
    # NOTE: I recommend that same experiments except seed should have same name (or group)
    #       for reporting statistical significance.

    model = DummyModel()

    for epoch in range(args.epochs):
        train_loss = model.train()
        valid_loss = model.valid()

        print("epoch: {}/{}, train_loss: {:.5f}, valid_loss: {:.5f}".format(
            epoch + 1,
            args.epochs,
            train_loss,
            valid_loss,
        ))

        # TODO: (2) upload "train_loss" and "valid_loss" per step
        # ref: https://docs.wandb.com/library/log
        # NOTE: image, audio, text, ..., can be uploaded by the integrated way!

    time.sleep(1)  # prevent output overwrite by wandb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()

    main(args)
