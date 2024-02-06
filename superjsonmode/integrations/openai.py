from openai import OpenAI
import os

from superjsonmode.data.parser import insert_into_path
from superjsonmode.data.prompts import DEFAULT_PROMPT, SINGLE_PASS_PROMPT
from superjsonmode.integrations.base_integration import BaseIntegration
from pydantic import BaseModel

from typing import Any, List, Dict, Optional


class StructuredOpenAIModel(BaseIntegration):
    def __init__(self, api_key=None, model="gpt-3.5-turbo-instruct"):
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            if "OPENAI_API_KEY" in os.environ:
                self.client = OpenAI()
            else:
                raise EnvironmentError(
                    "Please set the OPENAI_API_KEY environment variable to your API key."
                )
        self.model = model

    def generate(
        self,
        prompt: str,
        extraction_prompt_template: str = DEFAULT_PROMPT,
        schema: str or BaseModel = None,
        batch_size: int = 4,
        # max_new_tokens needs to be large enough to fit the largest value in the schema
        max_new_tokens: int = 20,
        stop: list[str] = ["\n"],
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

            results = self.client.completions.create(
                model=self.model,
                prompt=prompts,
                max_tokens=max_new_tokens,
                stop=stop,
                **kwargs,
            )
            outputs = [result.text for result in results.choices]

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

        result = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_new_tokens,
            **kwargs,
        )
        output = result.choices[0].text
        return output
