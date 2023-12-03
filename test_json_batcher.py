from madlibs import *


def generate_prompt(passage, schema):
    user_message = f"""{passage}
    From the above passage, extract the following schema: {schema}

    Only output JSON with the allowed types."""
    
    prompt = f"""<s><<SYS>>You only respond in JSON. You do not add text before. You do not add text after. Only JSON. <</SYS>>[INST] {user_message} [/INST]"""
    return prompt

batcher = JSONGenerator('jsonbench.jsonl')
data, schemas  = batcher.get_dataset(generate_prompt)