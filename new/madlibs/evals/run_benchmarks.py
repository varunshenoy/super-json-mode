import json
import time
from tqdm import tqdm
from prettytable import PrettyTable
import numpy as np
from enum import Enum

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import sys

from madlibs.integrations.transformers import StructuredOutputForModel
from pydantic import BaseModel
from eval_json import StructuredDatasetEvaluator


class Backend(Enum):
    TRANSFORMERS = "transformers"
    VLLM = "vllm"


def run_json_benchmark(
    model_id: str, backend: Backend, benchmark_file: str, **generation_kwargs
):
    device = "cuda"

    if backend == Backend.TRANSFORMERS:
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Create a structured output object
        structured_model = StructuredOutputForModel(model, tokenizer)

        evaluator = StructuredDatasetEvaluator(benchmark_file)

        evaluator.run(structured_model)

    elif backend == Backend.VLLM:
        structured_model = StructuredVLLMModel(model_id)


prompt = """NVIDIA Announces Financial Results for Third Quarter Fiscal 2024
November 21, 2023
Record revenue of $18.12 billion, up 34% from Q2, up 206% from year ago
Record Data Center revenue of $14.51 billion, up 41% from Q2, up 279% from year ago
NVIDIA (NASDAQ: NVDA) today reported revenue for the third quarter ended October 29, 2023, of $18.12 billion, up 206% from a year ago and up 34% from the previous quarter.

GAAP earnings per diluted share for the quarter were $3.71, up more than 12x from a year ago and up 50% from the previous quarter. Non-GAAP earnings per diluted share were $4.02, up nearly 6x from a year ago and up 49% from the previous quarter.

“Our strong growth reflects the broad industry platform transition from general-purpose to accelerated computing and generative AI,” said Jensen Huang, founder and CEO of NVIDIA.

“Large language model startups, consumer internet companies and global cloud service providers were the first movers, and the next waves are starting to build. Nations and regional CSPs are investing in AI clouds to serve local demand, enterprise software companies are adding AI copilots and assistants to their platforms, and enterprises are creating custom AI to automate the world’s largest industries.

“NVIDIA GPUs, CPUs, networking, AI foundry services and NVIDIA AI Enterprise software are all growth engines in full throttle. The era of generative AI is taking off,” he said.

NVIDIA will pay its next quarterly cash dividend of $0.04 per share on December 28, 2023, to all shareholders of record on December 6, 2023."""

run_json_benchmark(
    "mistralai/Mistral-7B-Instruct-v0.1",
    Backend.TRANSFORMERS,
    "../../benchmark/jsonbench.jsonl",
)
