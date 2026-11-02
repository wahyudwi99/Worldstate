"""Command-line entry point: rollout."""
from _bootstrap import run
from world_models.commands import rollout_main

if __name__ == "__main__":
    run(rollout_main)
