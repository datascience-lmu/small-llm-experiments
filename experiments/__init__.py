from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer

import itertools
import random
import string
import math
import ast
import re


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
        maybe_list = ast.literal_eval(last)
        if type(maybe_list) == list:
            return maybe_list


def check_response_contains_expected(response: str, expected: list[int]) -> bool:
    extracted = extract_list_from_response(response)
    return extracted == expected


class SmolChatbot:
    def __init__(
        self, model_name: str = "HuggingFaceTB/SmolLM3-3B", system_prompt: str = ""
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history: list[dict[str, str]] = []

        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def reset(self, system_prompt: str = ""):
        self.history = []

        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def __call__(self, user_input: str, enable_thinking=True):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        inputs = self.tokenizer(text, return_tensors="pt")
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


class QwenChatbot:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", system_prompt: str = ""):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history: list[dict[str, str]] = []

        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def reset(self, system_prompt: str = ""):
        self.history = []

        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def __call__(self, user_input: str, enable_thinking=True):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][
            len(inputs.input_ids[0]) :
        ].tolist()

        try:
            # rindex finding 151668 (</think>)
            split_index = len(response_ids) - response_ids[::-1].index(151668)
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


class GPTChatbot:
    def __init__(self, model_name: str = "openai/gpt-oss-20b", system_prompt: str = ""):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history: list[dict[str, str]] = []

        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def reset(self, system_prompt: str = ""):
        self.history = []

        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def __call__(self, user_input: str, enable_thinking=True):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        inputs = self.tokenizer(text, return_tensors="pt")
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
