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
        dag: Optional[Dict[str, List[str]]] = None,
    ):
        if isinstance(schema, BaseModel):
            self.schema = convert_schema_from_pydantic(schema)
        else:
            self.schema = schema

        self.batch_size = batch_size
        self.dag = dag

        self.items = list(self.processing_items(self.schema))
        if self.dag:
            self.items = self.order_items_by_dag(self.items, self.dag)
        self.batches = self.create_batches(self.items)

    def order_items_by_dag(
        self, items: List[SchemaItem], dag: Dict[str, List[str]]
    ) -> List[SchemaItem]:
        """Order items based on the given directed acyclic graph (DAG)."""
        ordered_items = []
        for node, edges in dag.items():
            node_items = [item for item in items if item.path[-1] == node]
            edge_items = [item for item in items if item.path[-1] in edges]
            ordered_items.extend(node_items + edge_items)
        return ordered_items

    def processing_items(
        self, schema: Dict[str, Any], path: List[Union[str, int]] = []
    ) -> Generator[SchemaItem, None, None]:
        """Generator function yielding schema items, handling 'object' and 'array' types."""
        if schema["type"] == "object":
            for k, v in schema["properties"].items():
                yield from self.processing_items(v, path + [k])
        else:
            yield SchemaItem(path=path, type_=schema["type"])

    def create_batches(self, items: List[SchemaItem]):
        """Create batches from given schema items ensuring dependencies are not in the same batch."""
        batches = []
        current_batch = []

        for item in items:
            if self.dag and item.path[-1] in self.dag and self.dag[item.path[-1]] != []:
                if current_batch != []:
                    batches.append(ProcessingBatch(items=current_batch))
                    current_batch = []
                batches.append(ProcessingBatch(items=[item]))
            elif len(current_batch) < self.batch_size:
                current_batch.append(item)
            else:
                batches.append(ProcessingBatch(items=current_batch))
                current_batch = [item]

        if current_batch:
            batches.append(ProcessingBatch(items=current_batch))

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
