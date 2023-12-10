import json
import time
from tqdm import tqdm
from prettytable import PrettyTable
import numpy as np
from enum import Enum

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import sys

from madlibs.integrations.transformers import StructuredOutputForModel

# from madlibs.integrations.vllm import StructuredVLLMModel
from pydantic import BaseModel
from madlibs.evals.eval_json import StructuredDatasetEvaluator


class Backend(Enum):
    TRANSFORMERS = "transformers"
    VLLM = "vllm"


class BenchmarkRunner:
    def __init__(self, model_id, backend: Backend):
        device = "cuda"
        self.evaluator = None
        if backend == Backend.TRANSFORMERS:
            model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.pad_token = tokenizer.eos_token

            # Create a structured output object
            self.model_to_benchmark = StructuredOutputForModel(model, tokenizer)

    def run_json_benchmark(
        self, benchmark_file: str, batch_size=4, **generation_kwargs
    ):
        self.evaluator = StructuredDatasetEvaluator(benchmark_file)
        out = self.evaluator.run(self.model_to_benchmark, batch_size=batch_size)

    def print_evals(self):
        self.evaluator.run_eval()


# run_json_benchmark(
#     "mistralai/Mistral-7B-Instruct-v0.1",
#     Backend.TRANSFORMERS,
#     "../../benchmark/jsonbench.jsonl",
# )
