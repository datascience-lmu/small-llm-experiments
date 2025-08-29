from time import sleep
import click


@click.group()
def cli():
    """A small collection of llm experiments"""
    pass


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

    click.echo("Loading model")
    qwen = experiments.QwenChatbot()

    iter = itertools.product(range(1, max_digits + 1), range(1, max_words + 1))

    res = {
        "number of digits": [],
        "number of words": [],
        "success count": [],
        "fail count": [],
    }

    click.echo("Running experiment")

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

    filename = "results/list" + time.strftime("%y-%m-%d--%H-%M-%S") + ".parquet"

    click.echo(dataset)

    dataset.write_parquet(filename)


@cli.command()
def stuff():
    import experiments

    gpt = experiments.GPTChatbot()

    print(gpt("Hello World")) 

if __name__ == "__main__":
    cli()
