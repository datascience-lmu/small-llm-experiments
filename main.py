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
@click.option("--q4", default=False)
@click.option("--q8", default=False)
def list(
    max_digits: int,
    max_words: int,
    samples: int,
    batch_size: int,
    step_size: int,
    thinking: bool,
    small: bool,
    q4: bool,
    q8: bool,
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

    qwen = experiments.QwenChatbot(model_name=model_name, q4=q4, q8=q8)
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
@click.option("--thinking", default=False)
@click.option("--small", default=False)
def noise_a(thinking: bool, small: bool):
    import experiments

    qwen = experiments.QwenChatbot(
        system_prompt='I will send you several messages. After each message only reply with "yes", "no" or "?". If you observe a complete question, only answer with "yes" or "no". If the question is not complete, answer with "?". The question might spread over several messages.'
    )

    print(qwen("Is", enable_thinking=thinking))
    print(qwen("Berlin", enable_thinking=thinking))
    print(qwen("a city", enable_thinking=thinking))

    print(qwen.history)


@cli.command()
@click.option("--thinking", default=False)
@click.option("--small", default=False)
def noise_b(thinking: bool, small: bool):
    import experiments

    qwen = experiments.QwenChatbot(system_prompt='Only answer with "yes" or "no".')

    print(qwen("Is Berlin a city?", enable_thinking=thinking))
    print(qwen("Is Berlin a in France?", enable_thinking=thinking))

    print(qwen.history)


if __name__ == "__main__":
    cli()
