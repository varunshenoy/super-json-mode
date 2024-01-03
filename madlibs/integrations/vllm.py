import vllm
from vllm import SamplingParams
from madlibs.data.parser import SchemaBatcher, SchemaItem, insert_into_path
from pydantic import BaseModel


DEFAULT_PROMPT = """Prompt: {prompt}

Based on the prompt, generate a value for the following key:

{key}: """

SINGLE_PASS_PROMPT = """[INST]{prompt}

Based on this excerpt, fill out the following schema:
{schema}
[/INST]"""

class VLLMConstrainedSamplingProcessor:
    # def __init__(self, max_new_tokens):
    #     self.max_new_tokens = max_new_tokens

    def __call__(self, token_ids, logits):
        # only allow integers and floats
        
        


class StructuredVLLMModel:
    def __init__(self, model_id):
        self.llm = vllm.LLM(model=model_id)

    def generate_prompt(
        self,
        prompt: str,
        batch_item: SchemaItem,
        extraction_prompt_template: str = DEFAULT_PROMPT,
    ):
        """Generate a prompt for a single item in a batch."""

        return extraction_prompt_template.format(
            prompt=prompt, key=batch_item.path[-1], type=batch_item.type_
        )

    def generate(
        self,
        prompt: str,
        extraction_prompt_template: str = DEFAULT_PROMPT,
        schema: str or BaseModel = None,
        batch_size: int = 4,
        max_new_tokens: int = 20,
        use_constrained_sampling=True,
        **kwargs,
    ):
        schema_batcher = SchemaBatcher(schema, batch_size=batch_size)
        batches = schema_batcher.batches

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
        max_new_tokens: int = 256,
        **kwargs,
    ):
        prompt = extraction_prompt_template.format(prompt=prompt, schema=schema)

        sampling_params = SamplingParams(**kwargs)
        sampling_params.max_tokens = max_new_tokens

        result = self.llm.generate(prompt, sampling_params=sampling_params)[0]
        output = result.outputs[0].text
        return output
