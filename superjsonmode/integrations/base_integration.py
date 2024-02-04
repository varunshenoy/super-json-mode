from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
from superjsonmode.data.parser import (
    SchemaBatcher,
    SchemaItem,
    insert_into_path,
    array_to_yaml,
)
from superjsonmode.data.prompts import DEFAULT_PROMPT, SINGLE_PASS_PROMPT
from pydantic import BaseModel


class BaseIntegration(ABC):
    def generate_prompt(
        self,
        prompt: str,
        batch_item: SchemaItem,
        extraction_prompt_template: str = DEFAULT_PROMPT,
    ):
        """Generate a prompt for a single item in a batch."""

        path = array_to_yaml(batch_item.path)
        key = batch_item.path[-1]
        return extraction_prompt_template.format(
            prompt=prompt, key=key, type=batch_item.type_, path=path
        )

    def generate_batches(
        self,
        schema: str or BaseModel,
        batch_size: int
    ):
        schema_batcher = SchemaBatcher(
            schema, batch_size=batch_size
        )
        batches = schema_batcher.batches
        return batches

    @abstractmethod
    def generate(
        self,
        prompt: str,
        extraction_prompt_template: str = DEFAULT_PROMPT,
        schema: str or BaseModel = None,
        batch_size: int = 4,
        # max_new_tokens needs to be large enough to fit the largest value in the schema
        max_new_tokens: int = 20,
        use_constrained_sampling=True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Abstract method for generating batched queries. Users implement this method for their specific integration.
        See `superjsonmode/integrations/vllm.py` for an example.
        """
        pass

    @abstractmethod
    def default_generate(
        self,
        prompt: str,
        extraction_prompt_template: str = SINGLE_PASS_PROMPT,
        schema: str or BaseModel = None,
        # max_new_tokens needs to be large enough to fit the filled-in schema
        max_new_tokens: int = 256,
        **kwargs,
    ) -> str:
        """
        Abstract method for a default generation process. Needs implementation in derived classes.
        """
        pass
