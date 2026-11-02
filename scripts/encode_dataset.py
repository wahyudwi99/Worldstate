"""Command-line entry point: encode dataset."""
from _bootstrap import run
from world_models.commands import encode_main

if __name__ == "__main__":
    run(encode_main)
