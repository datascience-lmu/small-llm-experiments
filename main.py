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
    import torch

    logger.info("Loading model")

    qwen = experiments.QwenChatbot(model_name="Qwen/Qwen3-0.6B")
    experiments.list_test(qwen, max_digits, max_words, samples, False)
    del qwen
    gc.collect()
    torch.cuda.empty_cache()

    qwen = experiments.QwenChatbot(model_name="Qwen/Qwen3-1.7B")
    experiments.list_test(qwen, max_digits, max_words, samples, False)
    del qwen
    gc.collect()
    torch.cuda.empty_cache()

    qwen = experiments.QwenChatbot(model_name="Qwen/Qwen3-4B")
    experiments.list_test(qwen, max_digits, max_words, samples, False)
    del qwen
    gc.collect()
    torch.cuda.empty_cache()

    qwen = experiments.QwenChatbot(model_name="Qwen/Qwen3-8B")
    experiments.list_test(qwen, max_digits, max_words, samples, False)
    del qwen
    gc.collect()
    torch.cuda.empty_cache()

    qwen = experiments.QwenChatbot(model_name="Qwen/Qwen3-14B")
    experiments.list_test(qwen, max_digits, max_words, samples, False)
    del qwen
    gc.collect()
    torch.cuda.empty_cache()

    qwen = experiments.QwenChatbot(model_name="Qwen/Qwen3-0.6B")
    experiments.list_test(qwen, max_digits, max_words, samples, True)
    del qwen
    gc.collect()
    torch.cuda.empty_cache()

    qwen = experiments.QwenChatbot(model_name="Qwen/Qwen3-1.7B")
    experiments.list_test(qwen, max_digits, max_words, samples, True)
    del qwen
    gc.collect()
    torch.cuda.empty_cache()

    qwen = experiments.QwenChatbot(model_name="Qwen/Qwen3-4B")
    experiments.list_test(qwen, max_digits, max_words, samples, True)
    del qwen
    gc.collect()
    torch.cuda.empty_cache()

    qwen = experiments.QwenChatbot(model_name="Qwen/Qwen3-8B")
    experiments.list_test(qwen, max_digits, max_words, samples, True)
    del qwen
    gc.collect()
    torch.cuda.empty_cache()

    qwen = experiments.QwenChatbot(model_name="Qwen/Qwen3-14B")
    experiments.list_test(qwen, max_digits, max_words, samples, True)
    del qwen
    gc.collect()
    torch.cuda.empty_cache()

    # gpt = experiments.GPTChatbot()
    # experiments.list_test(gpt, max_digits, max_words, samples, False)
    # del gpt
    # gc.collect()
    # torch.cuda.empty_cache()

    # smol = experiments.SmolChatbot()
    # experiments.list_test(smol, max_digits, max_words, samples, False)
    # del smol
    # gc.collect()
    # torch.cuda.empty_cache()

    # # ---

    # qwen = experiments.QwenChatbot()
    # experiments.list_test(qwen, max_digits, max_words, samples, True)
    # del qwen
    # gc.collect()
    # torch.cuda.empty_cache()

    # qwen = experiments.QwenChatbot(model_name="Qwen/Qwen3-14B")
    # experiments.list_test(qwen, max_digits, max_words, samples, True)
    # del qwen
    # gc.collect()
    # torch.cuda.empty_cache()

    # gpt = experiments.GPTChatbot()
    # experiments.list_test(gpt, max_digits, max_words, samples, True)
    # del gpt
    # gc.collect()
    # torch.cuda.empty_cache()

    # smol = experiments.SmolChatbot()
    # experiments.list_test(smol, max_digits, max_words, samples, True)
    # del smol
    # gc.collect()
    # torch.cuda.empty_cache()


@cli.command()
def stuff():
    import experiments

    qwen = experiments.QwenChatbot()

    print(qwen("Keep this message for later: abc"))

    print(qwen("What was the previous message?"))


if __name__ == "__main__":
    cli()
