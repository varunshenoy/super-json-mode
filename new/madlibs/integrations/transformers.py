from transformers import PreTrainedModel, PreTrainedTokenizerBase
import torch
from madlibs.data.parser import SchemaBatcher, SchemaItem, insert_into_path
from pydantic import BaseModel

DEFAULT_PROMPT = """Prompt: {prompt}

Based on the prompt, generate a value for the following key:

{key}: """


class StructuredOutputForModel:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        self.model = model
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        # This method will be called if the attribute/method "name" is not found
        # in this object, in which case we try to access the attribute from the model.
        return getattr(self.model, name)

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
        max_new_tokens: int = 256,
        do_sample: bool = True,
        **kwargs,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        schema_batcher = SchemaBatcher(schema, batch_size=batch_size)
        batches = schema_batcher.batches
        output_json = {}

        for batch in batches:
            print("running batch")
            # 1. create batch of prompts
            prompts = [
                self.generate_prompt(
                    prompt, item, extraction_prompt_template=extraction_prompt_template
                )
                for item in batch.items
            ]
            # print(prompts)

            # 2. encode batch of prompts
            embeds = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
                device
            )

            # 3. generate batch of outputs
            prediction = self.model.generate(
                **embeds, max_new_tokens=max_new_tokens, do_sample=do_sample, **kwargs
            )

            # 4. decode batch of outputs
            outputs = self.tokenizer.batch_decode(
                prediction[:, embeds["input_ids"].shape[1] :]
            )

            # 5. insert outputs into schema
            for item, output in zip(batch.items, outputs):
                insert_into_path(output_json, item.path, output.strip())

            print(output_json)

        return output_json
