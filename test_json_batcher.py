from madlibs import *
from madlibs.utils.postprocessing import build_json

OUTPUT_FNAME = "/Users/alexderhacobian/Downloads/batch-size=2_max-length=512_jsonbench_outputs (2).txt"

def generate_prompt(passage, schema):
    user_message = f"""{passage}
    From the above passage, extract the following schema: {schema}

    Only output JSON with the allowed types."""
    
    prompt = f"""<s><<SYS>>You only respond in JSON. You do not add text before. You do not add text after. Only JSON. <</SYS>>[INST] {user_message} [/INST]"""
    return prompt

batcher = JSONBatcher('example-jsons/jsonbench.jsonl')
data, schemas, original_properties, prompt_ids = batcher.get_dataset(generate_prompt)

with open(OUTPUT_FNAME, "r") as f:
    cleaned_outputs = f.readlines()
    build_json(cleaned_outputs, original_properties, prompt_ids)