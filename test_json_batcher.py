from json_batcher import JSONBatcher

def generate_prompt(passage, schema):
        user_message = f"""{passage}
    
From the above passage, extract the following schema:
{schema}

Only output JSON with the allowed types."""
        prompt = f"""<s><<SYS>>You only respond in JSON. You do not add text before. You do not add text after. Only JSON.<</SYS>>[INST] {user_message} [/INST]"""
        return prompt

batcher = JSONBatcher('jsonbench.jsonl')
print(batcher.get_batches(generate_prompt))