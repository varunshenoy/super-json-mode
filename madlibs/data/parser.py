from pydantic import BaseModel
from typing import Optional, Dict, Any, Union, List, Generator
from .utils import convert_schema_from_pydantic


class SchemaItem(BaseModel):
    """Data model representing a single item in a JSON schema."""

    path: List[Union[str, int]]
    type_: str
    generated_value: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None


class ProcessingBatch(BaseModel):
    """Data model representing a batch of items to be processed."""

    items: List[SchemaItem]


class SchemaBatcher:
    def __init__(
        self,
        schema: Union[BaseModel, Dict[str, Any]],
        batch_size: int,
    ):
        if isinstance(schema, dict):
            self.schema = schema
        elif issubclass(schema, BaseModel): 
            self.schema = convert_schema_from_pydantic(schema)
        else:
            raise ValueError("Schema is not a Pydantic object or a JSON representation of one.")

        self.batch_size = batch_size

        self.items = list(self.processing_items(self.schema))
        self.batches = self.create_batches(self.items, self.batch_size)

    def processing_items(
        self, schema: Dict[str, Any], path: List[Union[str, int]] = []
    ) -> Generator[SchemaItem, None, None]:
        """Generator function yielding schema items."""

        # TODO: attach metadata to schema items if needed
        if schema["type"] == "object":
            for k, v in schema["properties"].items():
                yield from self.processing_items(v, path + [k])
        else:
            yield SchemaItem(path=path, type_=schema["type"])

    def create_batches(self, items: List[SchemaItem], batch_size: int):
        """Create batches from given schema items."""
        batches = [
            ProcessingBatch(items=items[i : i + batch_size])
            for i in range(0, len(items), batch_size)
        ]
        return batches


def insert_into_path(root: Dict, path: List[Union[str, int]], value: Any):
    """Insert value into nested dictionary at specified path."""
    for p in path[:-1]:
        # If p is an index of a list, ensure list is long enough
        if isinstance(p, int):
            while len(root) <= p:
                root.append(None)
            root = root[p]
        else:
            root = root.setdefault(p, {})
    root[path[-1]] = value

def array_to_yaml(keys):
    yaml_string = ""
    for i, key in enumerate(keys):
        yaml_string += "\t" * i + key + ":" + "\n"
    return yaml_string