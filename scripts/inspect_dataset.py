"""Command-line entry point: inspect dataset."""
from _bootstrap import run
from world_models.commands import inspect_main

if __name__ == "__main__":
    run(inspect_main)
