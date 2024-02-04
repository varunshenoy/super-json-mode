import vllm
from vllm import SamplingParams
from superjsonmode.data.parser import insert_into_path
from superjsonmode.data.prompts import DEFAULT_PROMPT, SINGLE_PASS_PROMPT
from superjsonmode.integrations.base_integration import BaseIntegration
from pydantic import BaseModel

from typing import Any, List, Dict, Optional


class StructuredVLLMModel(BaseIntegration):
    def __init__(self, model_id):
        self.llm = vllm.LLM(model=model_id)

    def generate(
        self,
        prompt: str,
        extraction_prompt_template: str = DEFAULT_PROMPT,
        schema: str or BaseModel = None,
        batch_size: int = 4,
        # max_new_tokens needs to be large enough to fit the largest value in the schema
        max_new_tokens: int = 20,
        use_constrained_sampling=True,
        **kwargs,
    ) -> Dict[str, Any]:
        batches = self.generate_batches(schema, batch_size=batch_size)

        output_json = {}

        for batch in batches:
            prompts = [
                self.generate_prompt(
                    prompt, item, extraction_prompt_template=extraction_prompt_template
                )
                for item in batch.items
            ]

            sampling_params = SamplingParams(**kwargs)
            sampling_params.max_tokens = max_new_tokens
            if use_constrained_sampling:
                # TODO: implement constrained sampling on the logits
                sampling_params.logits_processors = []

            results = self.llm.generate(prompts, sampling_params=sampling_params)
            outputs = [result.outputs[0].text for result in results]

            for item, output in zip(batch.items, outputs):
                insert_into_path(output_json, item.path, output.strip())

        return output_json

    def default_generate(
        self,
        prompt: str,
        extraction_prompt_template: str = SINGLE_PASS_PROMPT,
        schema: str or BaseModel = None,
        # max_new_tokens needs to be large enough to fit the filled-in schema
        max_new_tokens: int = 256,
        **kwargs,
    ) -> str:
        prompt = extraction_prompt_template.format(prompt=prompt, schema=schema)

        sampling_params = SamplingParams(**kwargs)
        sampling_params.max_tokens = max_new_tokens

        result = self.llm.generate(prompt, sampling_params=sampling_params)[0]
        output = result.outputs[0].text
        return output
