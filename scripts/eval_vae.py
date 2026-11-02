"""Command-line entry point: eval vae."""
from _bootstrap import run
from world_models.commands import eval_vae_main

if __name__ == "__main__":
    run(eval_vae_main)
