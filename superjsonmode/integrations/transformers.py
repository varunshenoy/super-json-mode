from transformers import PreTrainedModel, PreTrainedTokenizerBase
import torch
from superjsonmode.data.parser import SchemaBatcher, SchemaItem, insert_into_path
from superjsonmode.integrations.base_integration import BaseIntegration
from superjsonmode.data.prompts import DEFAULT_PROMPT, SINGLE_PASS_PROMPT
from pydantic import BaseModel


class StructuredOutputForModel(BaseIntegration):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        # This method will be called if the attribute/method "name" is not found
        # in this object, in which case we try to access the attribute from the model.
        return getattr(self.model, name)

    def generate(
        self,
        prompt: str,
        extraction_prompt_template: str = DEFAULT_PROMPT,
        schema: str or BaseModel = None,
        batch_size: int = 4,
        max_new_tokens: int = 20,
        do_sample: bool = True,
        **kwargs,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batches = self.generate_batches(schema, batch_size=batch_size)
        output_json = {}
        filler_tokens = ["</s>", "'", '"']

        for batch in batches:
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
                **embeds,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

            # 4. decode batch of outputs
            outputs = self.tokenizer.batch_decode(
                prediction[:, embeds["input_ids"].shape[1] :]
            )

            # 5. insert outputs into schema
            for item, output in zip(batch.items, outputs):
                for tok in filler_tokens:
                    output = output.replace(tok, "")
                insert_into_path(output_json, item.path, output.strip())

            # print(output_json)

        return output_json

    def default_generate(
        self,
        prompt: str,
        extraction_prompt_template: str = SINGLE_PASS_PROMPT,
        schema: str or BaseModel = None,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        **kwargs,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        prompt = extraction_prompt_template.format(prompt=prompt, schema=schema)
        embeds = self.tokenizer(prompt, return_tensors="pt").to(device)
        filler_tokens = ["</s>", "'", '"']
        prediction = self.model.generate(
            **embeds,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        output = self.tokenizer.batch_decode(
            prediction[:, embeds["input_ids"].shape[1] :]
        )[0]
        for tok in filler_tokens:
            output = output.replace(tok, "")

        print(output)
        return output
