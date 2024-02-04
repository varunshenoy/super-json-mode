import json
from prettytable import PrettyTable
import time


def load_dataset(dataset_file):
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f.readlines()]
    return dataset


superjsonmode_prompt_template = """[INST]{prompt}

Based on this excerpt, extract the correct value for "{key}". The answer is in the excerpt. Just print a short answer and then stop.

{key}: [/INST]"""

default_prompt_template = """[INST]{prompt}

Based on this excerpt, fill out the following schema in valid JSON:
{schema}

Do not just copy over the types. Retain the structure of the schema and fill in the appropriate values. Don't provide type values, just provide the actual values.

Use double quotes for the JSON.
[/INST]"""


class StructuredDatasetEvaluator:
    def __init__(self, dataset_file):
        self.dataset = load_dataset(dataset_file)

    def run(self, engine, run_batching=True, batch_size=4, **sampling_params):
        self.outputs = []
        self.run_times = []
        self.schemas = []
        self.batch_size = batch_size

        start_time = time.time()

        for sample in self.dataset:
            passage = sample["passage"]
            schema = sample["schema"]

            start_time = time.time()
            if run_batching:
                output = engine.generate(
                    passage,
                    extraction_prompt_template=superjsonmode_prompt_template,
                    schema=schema,
                    batch_size=batch_size,
                )
            else:
                output = engine.default_generate(
                    passage,
                    schema=schema,
                    extraction_prompt_template=default_prompt_template,
                )

            time_taken = round(time.time() - start_time, 3)
            self.run_times.append(time_taken)
            self.outputs.append(output)
            self.schemas.append(schema)

        return self.outputs, self.run_times

    def convert_schema_to_jsonformer_format(self, schema):
      # Convert to JSONFormer-compatible schema for schema checking
      jsonformer_schema = {}
      jsonformer_schema["type"] = "object"
      jsonformer_schema["properties"] = {}
      for key in schema.keys():
        if type(schema[key]) is dict:
            jsonformer_schema["properties"][key] = self.convert_schema_to_jsonformer_format(schema[key])
        else:
          jsonformer_schema["properties"][key] = {"type" : schema[key]}

      return jsonformer_schema

    def has_matching_schema(self, output, target):
        # Checks if JSON objects have matching schemas
        if isinstance(output, dict) and isinstance(target, dict):
            for key in target:
                if key not in output or not self.has_matching_schema(
                    output[key], target[key]
                ):
                    return False
            return True
        return isinstance(output, type(target))

    def print_evals(self, evals):
        # Prints evaluations in a PrettyTable
        table = PrettyTable()

        # print(evals)

        table.field_names = [
            "Valid (✅/❌)",
            "Matches Schema (✅/❌)",
            "Batch Size",
            "Time (s)",
            "Error",
        ]
        for eval in evals:
            table.add_row(
                [
                    eval["is_valid"],
                    eval["matches_schema"],
                    eval["batch_size"],
                    eval["time_taken"],
                    eval["error_type"],
                ]
            )

        print(table)

    def generate_eval(self, output, run_time, schema):
        # Generates a single evaluation
        evaluation = {
            "generation": output,
            "time_taken": run_time,
            "batch_size": self.batch_size,
            "is_valid": False,
            "matches_schema": False,
            "error_type": "unknown",
        }

        try:
            # Attempt to parse the generation as JSON
            if isinstance(evaluation["generation"], str):
                json_result = json.loads(evaluation["generation"])
            else:
                json_result = evaluation["generation"]
            evaluation["is_valid"] = True

            # Check if JSON result matches schema
            jsonformer_json_result = self.convert_schema_to_jsonformer_format(json_result)
            evaluation["matches_schema"] = self.has_matching_schema(jsonformer_json_result, schema)
            evaluation["error_type"] = (
                None if evaluation["matches_schema"] else "schema_mismatch"
            )

        except ValueError:
            evaluation["error_type"] = "invalid_json"

        return evaluation

    def run_eval(self):
        self.evals = []

        for output, run_time, schema in zip(self.outputs, self.run_times, self.schemas):
            eval = self.generate_eval(output, run_time, schema)
            self.evals.append(eval)

        self.print_evals(self.evals)
