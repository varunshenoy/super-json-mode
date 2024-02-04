# Default prompts

DEFAULT_PROMPT = """Prompt: {prompt}

Based on the prompt, generate a value for the following key. The value should be a {type}:

{key}: """

SINGLE_PASS_PROMPT = """[INST]{prompt}

Based on this excerpt, fill out the following schema:
{schema}
[/INST]"""
