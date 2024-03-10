import torch
from superjsonmode.data.parser import SchemaBatcher, SchemaItem, insert_into_path
from superjsonmode.integrations.base_integration import BaseIntegration
from superjsonmode.data.prompts import DEFAULT_PROMPT, SINGLE_PASS_PROMPT
from pydantic import BaseModel
import sglang as sgl
from enum import Enum


class RegexTypes(Enum):
    INTEGER = r"-?\d+"
    FLOAT = r"-?\d+(\.\d+)?"
    STRING = r"[\w\s]+"
    MONEY = r"(\$)(\d{1,3}(\,\d{3})*|(\d+))(\.\d{2})?"
    DATE = r"(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])-([0-9]{4})"
    BOOL = r"(?:(Tru|Fals)e"


@sgl.function
def kv_generate(s, prompt, regex, max_new_tokens=20):
    s += prompt + sgl.gen("answer", regex=regex, max_tokens=max_new_tokens)


class SGLStructuredOutputForModel(BaseIntegration):
    def __init__(self, backend_url="http://localhost:30000"):
        sgl.set_default_backend(sgl.RuntimeEndpoint(backend_url))

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
        **kwargs,
    ):
        batches = self.generate_batches(schema, batch_size=batch_size)
        output_json = {}
        filler_tokens = ["</s>", "'", '"']

        for batch in batches:
            prompts = []
            for item in batch.items:
                prompt = self.generate_prompt(
                    prompt,
                    item,
                    extraction_prompt_template=extraction_prompt_template,
                )

                regex = None
                if item.type_ == "number" or item.type_ == "integer":
                    regex = RegexTypes.INTEGER.value
                elif item.type_ == "float":
                    regex = RegexTypes.FLOAT.value
                elif item.type_ == "boolean":
                    regex = RegexTypes.BOOL.value
                if item.pattern is not None:
                    regex = item.pattern

                prompts.append(
                    {"prompt": prompt, "regex": regex, "max_new_tokens": max_new_tokens}
                )
            states = kv_generate.run_batch(prompts, progress_bar=True)
            outputs = []
            for i, state in enumerate(states):
                generated_output = (
                    state.text().replace(prompts[i]["prompt"], "").strip()
                )
                outputs.append(generated_output)

            for item, output in zip(batch.items, outputs):
                for tok in filler_tokens:
                    output = output.replace(tok, "")
                insert_into_path(output_json, item.path, output.strip())

        return output_json

    def default_generate(self, **kwargs):
        raise NotImplementedError("This method is not implemented yet.")
