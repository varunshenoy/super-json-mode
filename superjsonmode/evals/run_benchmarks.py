from enum import Enum

from transformers import AutoTokenizer, AutoModelForCausalLM

from superjsonmode.integrations.transformers import StructuredOutputForModel
from superjsonmode.integrations.vllm import StructuredVLLMModel
from superjsonmode.evals.eval_json import StructuredDatasetEvaluator


class Backend(Enum):
    TRANSFORMERS = "transformers"
    VLLM = "vllm"

class BenchmarkRunner:
    def __init__(self, model_id, backend: Backend):
        device = "cuda"
        self.evaluator = None
        self.backend = backend
        if backend == Backend.TRANSFORMERS:
            model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.pad_token = tokenizer.eos_token

            # Create a structured output object
            self.model_to_benchmark = StructuredOutputForModel(model, tokenizer)
        elif backend == Backend.VLLM:
            self.model_to_benchmark = StructuredVLLMModel(model_id)

    def run_json_benchmark(
        self, benchmark_file: str, batch_size=4, run_batching=True, **generation_kwargs
    ):
        self.evaluator = StructuredDatasetEvaluator(benchmark_file)
        print(run_batching)
        out = self.evaluator.run(
            self.model_to_benchmark, batch_size=batch_size, run_batching=run_batching
        )

    def print_evals(self):
        self.evaluator.run_eval()
