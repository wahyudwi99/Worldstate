"""Command-line entry point: train mdn rnn."""
from _bootstrap import run
from world_models.commands import train_mdn_main

if __name__ == "__main__":
    run(train_mdn_main)
