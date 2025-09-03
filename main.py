import click
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@click.group()
def cli():
    """A small collection of llm experiments"""
    logging.basicConfig()


@cli.command()
@click.option("--max-digits", default=3)
@click.option("--max-words", default=20)
@click.option("--samples", default=20)
def list(max_digits: int, max_words: int, samples: int):
    """Experiment to test if llm's can sort lists"""
    import itertools
    import time

    import experiments
    import polars as pl

    import gc

    logger.info("Loading model")

    qwen = experiments.QwenChatbot()
    experiments.list_test(qwen, max_digits, max_words, samples)
    del qwen
    gc.collect()

    qwen = experiments.QwenChatbot(model_name="Qwen/Qwen3-14B")
    experiments.list_test(qwen, max_digits, max_words, samples)
    del qwen
    gc.collect()

    gpt = experiments.GPTChatbot()
    experiments.list_test(gpt, max_digits, max_words, samples)
    del gpt
    gc.collect()

    smol = experiments.SmolChatbot()
    experiments.list_test(smol, max_digits, max_words, samples)
    del smol
    gc.collect()


@cli.command()
def stuff():
    import experiments

    qwen = experiments.QwenChatbot()

    print(qwen("Hello"))


if __name__ == "__main__":
    cli()
