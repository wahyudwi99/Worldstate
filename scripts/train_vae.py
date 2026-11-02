"""Command-line entry point: train vae."""
from _bootstrap import run
from world_models.commands import train_vae_main

if __name__ == "__main__":
    run(train_vae_main)
