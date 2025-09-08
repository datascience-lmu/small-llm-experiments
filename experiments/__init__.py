from copy import deepcopy
from typing import override
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import itertools
import logging
import random
import string
import math
import ast
import re

from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def add_characters(input: str, char: str = "x") -> str:
    return "".join(
        list(
            itertools.chain.from_iterable(
                zip(input, itertools.repeat(char, times=len(input)))
            )
        )
    )


# add random length
def add_words(input: str, word_length: int = 6) -> str:
    input_split = input.split(" ")
    junk = []
    for _ in range(len(input_split)):
        word = ""
        for _ in range(word_length):
            word += random.choice(string.ascii_letters)
        junk.append(word)
    return " ".join(list(itertools.chain.from_iterable(zip(input_split, junk))))


def split_across_messages(input: str, noise_length: int = 6) -> list[str]:
    input_split = input.split(" ")
    res = []

    for word in input_split:
        word += " "
        for _ in range(noise_length):
            word += random.choice(string.ascii_letters)

        res.append(word)

    return res


def scramble_word(input: str) -> str:
    words = input.split()
    res = []

    for word in words:
        if len(word) > 2:
            inner_letters = list(word)[1:-1]

            random.shuffle(inner_letters)

            new_word = word[0]
            new_word += "".join(inner_letters)
            new_word += word[-1]

            res.append(new_word)

        else:
            res.append(word)

    return " ".join(res)


def generate_list_prompt(length: int, digits: int) -> tuple[str, list[int]]:
    unsorted_list = [random.randint(0, 10**digits) for _ in range(length)]

    sorted_list = deepcopy(unsorted_list)
    sorted_list.sort()

    prompt = "Sort the list " + str(unsorted_list) + " from smallest to largest."

    return (prompt, sorted_list)


def extract_list_from_response(response: str) -> list[int] | None:
    occurrences = re.findall(r"\[[0-9,\,\ ]*\]", response)

    if len(occurrences) != 0:
        last = occurrences[-1]
        try:
            maybe_list = ast.literal_eval(last)
            if type(maybe_list) is list:
                return maybe_list
        except:  # noqa: E722
            return None


def check_response_contains_expected(response: str, expected: list[int]) -> bool:
    extracted = extract_list_from_response(response)
    return extracted == expected


class Chatbot(ABC):
    def reset(self, system_prompt: str = ""):
        self.history = []

        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    @abstractmethod
    def __call__(
        self, user_input: str | list[str], enable_thinking: bool
    ) -> str | list[str]:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class SmolChatbot(Chatbot):
    def __init__(
        self, model_name: str = "HuggingFaceTB/SmolLM3-3B", system_prompt: str = ""
    ):
        logger.info(f"Loading {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.name = "".join(x for x in model_name if x.isalnum())
        self.history: list[dict[str, str]] = []
        logger.debug(f"Loaded on device {self.model.device}")

        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def __call__(self, user_input: str, enable_thinking=True) -> str:
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][
            len(inputs.input_ids[0]) :
        ].tolist()

        try:
            # rindex finding 128003 (</think>)
            split_index = len(response_ids) - response_ids[::-1].index(128003)
        except ValueError:
            split_index = 0

        _reasoning_content = self.tokenizer.decode(
            response_ids[:split_index], skip_special_tokens=True
        ).strip("\n")
        content = self.tokenizer.decode(
            response_ids[split_index:], skip_special_tokens=True
        ).strip("\n")

        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return content

    def __str__(self) -> str:
        return self.name


class QwenChatbot(Chatbot):
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", system_prompt: str = ""):
        logger.info(f"Loading {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.name = "".join(x for x in model_name if x.isalnum())
        self.history: list[dict[str, str]] = []
        logger.debug(f"Loaded on device {self.model.device}")

        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def split_content(self, response_ids) -> tuple[str, str]:
        try:
            # rindex finding 151668 (</think>)
            split_index = len(response_ids) - response_ids[::-1].index(151668)
        except ValueError:
            split_index = 0

        reasoning_content = self.tokenizer.decode(
            response_ids[:split_index], skip_special_tokens=True
        ).strip("\n")
        content = self.tokenizer.decode(
            response_ids[split_index:], skip_special_tokens=True
        ).strip("\n")

        return (content, reasoning_content)

    def __call__(
        self, user_input: str | list[str], enable_thinking=True
    ) -> str | list[str]:
        if type(user_input) is list:
            messages = [
                self.history + [{"role": "user", "content": inp}] for inp in user_input
            ]

        elif type(user_input) is str:
            messages = self.history + [{"role": "user", "content": user_input}]

        else:
            raise ValueError

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        response_ids = self.model.generate(**inputs, max_new_tokens=32768)

        if type(user_input) is list:
            generated_texts = []
            for idx, _output in enumerate(response_ids):
                content, _reasoning_content = self.split_content(
                    response_ids[idx][len(inputs.input_ids[idx]) :].tolist()
                )

                generated_texts.append(content)

            return generated_texts

        elif type(user_input) is str:
            response_ids = response_ids[0][len(inputs.input_ids[0]) :].tolist()
            content, _reasoning_content = self.split_content(response_ids)
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": response})

            return content

        else:
            raise ValueError

    def __str__(self) -> str:
        return self.name


class GPTChatbot(Chatbot):
    def __init__(self, model_name: str = "openai/gpt-oss-20b", system_prompt: str = ""):
        logger.info(f"Loading {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.name = "".join(x for x in model_name if x.isalnum())
        self.history: list[dict[str, str]] = []
        logger.debug(f"Loaded on device {self.model.device}")

        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def __call__(self, user_input: str, enable_thinking=True) -> str:
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][
            len(inputs.input_ids[0]) :
        ].tolist()

        # try:
        #     # rindex finding 151668 (</think>)
        #     split_index = len(response_ids) - response_ids[::-1].index(151668)
        # except ValueError:
        #     split_index = 0

        # _reasoning_content = self.tokenizer.decode(
        #     response_ids[:split_index], skip_special_tokens=True
        # ).strip("\n")
        # content = self.tokenizer.decode(
        #     response_ids[split_index:], skip_special_tokens=True
        # ).strip("\n")

        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

    def __str__(self) -> str:
        return self.name


def list_test(
    model: Chatbot, max_digits: int, max_words: int, samples: int, enable_thinking=False
):
    """Experiment to test if llm's can sort lists"""
    import itertools
    import time

    import experiments
    import polars as pl

    iter = itertools.product(range(1, max_digits + 1), range(1, max_words + 1))

    res = {
        "thinking": [],
        "number of digits": [],
        "number of words": [],
        "success count": [],
        "fail count": [],
    }

    logger.info(f"Running experiment with {model}")

    filename = (
        "results/list--"
        + str(model)
        + "--"
        + time.strftime("%y-%m-%d--%H-%M-%S")
        + ".csv"
    )

    for number_of_digits, number_of_words in iter:
        success_count = 0
        fail_count = 0

        questions = []
        answers = []
        for _ in range(samples):
            question, answer = experiments.generate_list_prompt(
                number_of_words, number_of_digits
            )
            questions.append(question)
            answers.append(answer)

        # for _ in range(samples):
        #     prompt, expected = experiments.generate_list_prompt(
        #         number_of_words, number_of_digits
        #     )

        #     model.reset()
        #     response = model(
        #         prompt,
        #         enable_thinking=enable_thinking,
        #     )

        #     if experiments.check_response_contains_expected(response, expected):
        #         success_count += 1
        #     else:
        #         fail_count += 1

        model.reset()
        responses = model(questions, enable_thinking=enable_thinking)

        for response, answer in zip(responses, answers):
            if experiments.check_response_contains_expected(response, answer):
                success_count += 1
            else:
                fail_count += 1

        res["thinking"].append(enable_thinking)
        res["number of digits"].append(number_of_digits)
        res["number of words"].append(number_of_words)
        res["success count"].append(success_count)
        res["fail count"].append(fail_count)

        dataset = pl.DataFrame(res)
        Path("results").mkdir(parents=True, exist_ok=True)
        # Potential data corruption if program is cancelled during this
        dataset.write_csv(filename)

        logger.info(f"Finished {number_of_digits} digits, {number_of_words} words")

    logger.info(dataset)
