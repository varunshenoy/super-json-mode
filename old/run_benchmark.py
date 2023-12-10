import json
import time
from tqdm import tqdm
from prettytable import PrettyTable
import numpy as np


class JSONBenchmark:
    def __init__(self, dataset_file):
        self.dataset = self.load_dataset(dataset_file)

    def load_dataset(self, dataset_file):
        with open(dataset_file, "r") as f:
            dataset = [json.loads(line) for line in f.readlines()]
        return dataset

    def generate_prompt(self, passage, schema):
        user_message = f"""{passage}
    
From the above passage, extract the following schema:
{schema}

Only output JSON with the allowed types."""
        prompt = f"""<s><<SYS>>You only respond in JSON. You do not add text before. You do not add text after. Only JSON.<</SYS>>[INST] {user_message} [/INST]"""
        return prompt

    def run(self, generate, **kwargs):
        evals = []
        for data in tqdm(self.dataset):
            evaluation = {}

            prompt = self.generate_prompt(data["passage"], data["schema"])
            start_time = time.time()
            result = generate(prompt, **kwargs)[0]["generated_text"].strip()
            time_taken = round(time.time() - start_time, 3)

            evaluation["generation"] = result
            evaluation["time_taken"] = time_taken

            # check if result is valid JSON
            try:
                json_result = json.loads(result)
                evaluation["is_valid"] = True

                # check if result matches schema
                # JSON might have erroneous keys
                schema = data["extracted_data"]
                evaluation["matches_schema"] = json_result == schema
                evaluation["error_type"] = None
            except ValueError:
                evaluation["is_valid"] = False
                evaluation["matches_schema"] = False

                if result[0] != "{":
                    evaluation["error_type"] = "prefix"
                elif result[-1] != "}":
                    evaluation["error_type"] = "suffix"
                else:
                    evaluation["error_type"] = "invalid"

            evals.append(evaluation)

        return evals

    def print(self, results, show_generation=False):
        table = PrettyTable()

        # Define the table columns
        table.field_names = [
            "Valid (✅/❌)",
            "Matches Schema (✅/❌)",
            "Time (s)",
            "Error",
        ]
        if show_generation:
            table.add_column("Generation")

        valid_counter, schema_counter, total_time = 0, 0, 0

        for result in results:
            is_valid = "✅" if result["is_valid"] else "❌"
            matches_schema = "✅" if result["matches_schema"] else "❌"
            error_type = result["error_type"]

            valid_counter += result["is_valid"]
            schema_counter += result["matches_schema"]
            total_time += result["time_taken"]

            row = [is_valid, matches_schema, result["time_taken"], error_type]
            if show_generation:
                row.append(result["generation"])

            table.add_row(row)

        valid_accuracy = valid_counter / len(results)
        schema_accuracy = schema_counter / len(results)
        average_time = round(total_time / len(results), 3)

        table.add_row(["-", "-", "-", "-"])
        table.add_row(
            [
                f"Accuracy: {valid_accuracy}",
                f"Accuracy: {schema_accuracy}",
                f"Average: {average_time}",
                "-",
            ]
        )

        print(table)
