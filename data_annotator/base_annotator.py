from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datasets import Dataset
from utils.llm import LLMClient, OpenAIClientLLM
from data_annotator.prompt_manager import AnnotatePromptManager
import logging
logger = logging.getLogger(__name__)


class DataAnnotator(ABC):
    def __init__(
        self,
        llm_class: type[LLMClient] = None,
        annotation_columns=None,
        **llm_kwargs
    ):
        self.annotation_columns = ["llm_annotation"] if annotation_columns is None else annotation_columns
        self.llm = llm_class(**llm_kwargs) if llm_class else OpenAIClientLLM(**llm_kwargs)

    async def process_split(self, split_dataset: Dataset) -> Dict:
        """Process a single split asynchronously"""
        semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
        futures = [self.process_row(row, semaphore) for row in split_dataset]
        tuple_of_dict = await asyncio.gather(*futures)
        return {key: [row[key] for row in tuple_of_dict] for key in tuple_of_dict[0]}

    async def process_row(self, row: Dict, semaphore: asyncio.Semaphore) -> Dict:
        """Process a single example with rate limiting
           return: Dict of annotation_name(key): annotation_value
        """
        async with semaphore:
            processed = self.pre_process(row)
            processed = await self.a_call_llm(processed)
            return self.post_process(processed, row)

    @abstractmethod
    def pre_process(self, row: Dict) -> Dict:
        """Format the example into a prompt"""
        pass

    @abstractmethod
    async def a_call_llm(self, processed_dict: Dict) -> Dict:
        """Call LLM with formatted prompt"""
        pass

    @abstractmethod
    def post_process(self, processed: Dict, row: Dict) -> Dict:
        """Process LLM response into final format"""
        pass
