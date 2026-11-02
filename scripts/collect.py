"""Command-line entry point: collect."""
from _bootstrap import run
from world_models.commands import collect_main

if __name__ == "__main__":
    run(collect_main)
