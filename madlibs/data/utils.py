import json
from pydantic import BaseModel


def convert_schema_from_pydantic(schema: BaseModel, root_schema=None):
    # copy the entire schema at the root level only
    if root_schema is None:
        schema = schema.model_json_schema().copy()
        root_schema = schema

    if "$ref" in schema:
        # Resolve reference based on root schema and recursively convert it
        ref_path = schema["$ref"].split("/")[1:]  # split and remove empty root string
        ref_schema = root_schema
        for component in ref_path:
            ref_schema = ref_schema.get(component, {})
        return convert_schema_from_pydantic(ref_schema, root_schema)

    schema.pop("title", None)  # remove title keys
    schema.pop("required", None)  # remove title keys

    if schema.get("type") in {
        "integer",
        "float",
    }:  # convert integer/float types to number
        schema["type"] = "number"

    if "properties" in schema:
        for key, prop in schema["properties"].items():
            schema["properties"][key] = convert_schema_from_pydantic(prop, root_schema)

    elif "items" in schema:
        schema["items"] = convert_schema_from_pydantic(schema["items"], root_schema)

    schema.pop("$defs", None)
    return schema
