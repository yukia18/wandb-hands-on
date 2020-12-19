import os
import random
import argparse
import time

import numpy as np
import wandb

from constants import PROJECT
from model import DummyModel


def main(args):
    # fix seed
    if args.seed is not None:
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # init wandb run
    # ref: https://docs.wandb.com/library/init
    wandb.init(
        name=args.name,
        project=PROJECT,
        config=vars(args),
    )

    model = DummyModel(use_sota=args.use_sota)

    for epoch in range(args.epochs):
        train_loss = model.train()
        valid_loss = model.valid()

        outputs = {
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        }
        print("epoch: {}/{}, outputs: {}".format(
            epoch + 1,
            args.epochs,
            outputs
        ))

        # you can upload score per step
        # image, audio, text, ..., can be uploaded by the integrated way!
        # ref: https://docs.wandb.com/library/log
        wandb.log(outputs, step=epoch)

    time.sleep(1)  # prevent output overwrite by wandb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_sota", action="store_true")

    args = parser.parse_args()

    main(args)
