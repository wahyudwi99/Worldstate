"""Command-line entry point: eval mdn rnn."""
from _bootstrap import run
from world_models.commands import eval_mdn_main

if __name__ == "__main__":
    run(eval_mdn_main)
