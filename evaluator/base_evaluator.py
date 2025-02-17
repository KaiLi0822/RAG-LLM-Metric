from __future__ import annotations  # for pervious python version e.g. 3.9
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from datasets import Dataset

from utils.llm import LLMClient, OpenAIClientLLM
from .prompt_manager import EvaluationType
import asyncio


class RAGEvaluator(ABC):
    """Base class for evaluating RAG outputs using LLM-as-a-judge pattern."""

    def __init__(
        self,
        llm_class: type[LLMClient] = None,
        **llm_kwargs
    ):
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
            processed = self.pre_process_row(row)
            response = await self.a_call_llm(processed)
            return self.post_process_row(response, row)

    @abstractmethod
    def pre_process_row(self, row: Dict) -> Dict:
        """Preprocess row"""
        raise NotImplementedError

    @abstractmethod
    async def a_call_llm(self, processed: Dict) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def post_process_row(self, processed: Dict, row) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def pre_process(self, question: str | List[str], context: str | List[str], answer: str | List[str],
                    **kwargs) -> Any:
        """
        Prepare and format the evaluation input.
        
        Args:
            question: User question/query
            context: Retrieved context used for generation
            answer: Generated answer to evaluate
            kwargs: Additional template parameters (golded_answer)
            
        Returns:
            Processed data ready for LLM evaluation
        """
        pass

    @abstractmethod
    def call_llm(self, processed_data: Any) -> str:
        """
        Execute the LLM call with the processed evaluation prompt.
        
        Args:
            processed_data: Formatted evaluation prompt from pre_process
            
        Returns:
            Raw LLM response string
        """
        pass

    @abstractmethod
    def post_process(self, llm_response: str, **kwargs) -> Dict[str, float]:
        """
        Convert LLM response into evaluation scores.
        
        Args:
            llm_response: Raw response string from LLM
            
        Returns:
            Dictionary of evaluation metrics and scores
        """
        pass

    def evaluate(self, answer: str | List[str] = None, question: str | List[str] = None,
                 context: str | List[str] = None, **kwargs) -> Dict:
        """
        Main evaluation workflow.
        
        Args:
            question: User question/query
            context: Retrieved context used for generation
            answer: Generated answer to evaluate
            kwargs: Additional template parameters (golded_answer)
            
        Returns:
            Dictionary of evaluation metrics and scores
        """
        processed_data = self.pre_process(question, context, answer, **kwargs)
        llm_response = self.call_llm(processed_data)
        return self.post_process(llm_response)
