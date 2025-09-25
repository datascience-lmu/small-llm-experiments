# Small LLM Experiments

## Installation

The project uses the [uv](https://docs.astral.sh/uv/) package manager. For detailed instructions see the [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

On macOS and Linux run
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

and on Windows
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Clone Repository

```shell
git clone https://github.com/datascience-lmu/small-llm-experiments.git
cd small-llm-experiments
```

## Install Python Packages

To install all required packages run
```shell
uv sync
```

## Run Experiments

```shell
uv run main.py
```

for example, to run the list sorting experiment with
- Lists up to length 50 (Words)
- 8 samples per experiment
- 8 parallel experiments (batch size)
- no chain of thought (thinking)
- large models (Qwen-14B)
- 4bit quantization
you can run
```shell
uv run main.py list --max-words 50 --samples 8 --batch-size 8 --step-size 10 --thinking false --small true --q4 true
```