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
@click.option("--batch-size", default=8)
@click.option("--step-size", default=1)
@click.option("--thinking", default=False)
@click.option("--small", default=False)
def list(
    max_digits: int,
    max_words: int,
    samples: int,
    batch_size: int,
    step_size: int,
    thinking: bool,
    small: bool,
):
    """Experiment to test if llm's can sort lists"""
    import experiments

    import gc
    import torch

    if small:
        model_name = "Qwen/Qwen3-0.6B"
    else:
        model_name = "Qwen/Qwen3-14B"

    logger.info(f"Loading model {model_name}")

    qwen = experiments.QwenChatbot(model_name=model_name)
    experiments.list_test(
        qwen, max_digits, max_words, samples, batch_size, step_size, thinking
    )
    del qwen
    gc.collect()
    torch.cuda.empty_cache()


@cli.command()
@click.option("--samples", default=20)
def noise(samples: int):
    import experiments

    logger.info("Loading model")

    qwen = experiments.QwenChatbot(model_name="Qwen/Qwen3-0.6B")
    experiments.noise_test(qwen, samples, False)
    del qwen


@cli.command()
def stuff_a():
    import experiments

    # qwen = experiments.QwenChatbot(
    #     system_prompt='I will send you several messages. After each message only reply with "yes", "no" or "?". If you observe a complete question, only answer with "yes" or "no". If the question is not complete, answer with "?". The question might spread over several messages.'
    # )

    # print(qwen("Is", enable_thinking=False))
    # print(qwen("Berlin", enable_thinking=False))
    # print(qwen("a city", enable_thinking=False))

    # print(qwen.history)

    qwen = experiments.QwenChatbot(system_prompt='Only answer with "yes" or "no".')

    print(qwen("Is Berlin a city?", enable_thinking=True))
    print(qwen("Is Berlin a in France?", enable_thinking=True))


if __name__ == "__main__":
    cli()
