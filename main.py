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

    logger.info("Loading model")
    qwen = experiments.QwenChatbot()

    iter = itertools.product(range(1, max_digits + 1), range(1, max_words + 1))

    res = {
        "number of digits": [],
        "number of words": [],
        "success count": [],
        "fail count": [],
    }

    logger.info("Running experiment")

    filename = "results/list--" + time.strftime("%y-%m-%d--%H-%M-%S") + ".parquet"

    with click.progressbar(iter, length=max_digits * max_words, show_pos=True) as bar:
        for number_of_digits, number_of_words in bar:
            success_count = 0
            fail_count = 0
            for _ in range(samples):
                prompt, expected = experiments.generate_list_prompt(
                    number_of_words, number_of_digits
                )

                qwen.reset()
                response = qwen(
                    prompt,
                    enable_thinking=False,
                )

                if experiments.check_response_contains_expected(response, expected):
                    success_count += 1
                else:
                    fail_count += 1

            res["number of digits"].append(number_of_digits)
            res["number of words"].append(number_of_words)
            res["success count"].append(success_count)
            res["fail count"].append(fail_count)

            dataset = pl.DataFrame(res)
            # Potential data corruption if program is cancelled during this
            dataset.write_parquet(filename)

    logger.info(dataset)


@cli.command()
def stuff():
    import experiments

    qwen = experiments.QwenChatbot()

    print(qwen("Hello"))


if __name__ == "__main__":
    cli()
