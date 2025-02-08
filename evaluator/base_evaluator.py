from abc import ABC, abstractmethod
from typing import Any, Dict
from .prompt_manager import EvaluationType

class RAGEvaluator(ABC):
    """Base class for evaluating RAG outputs using LLM-as-a-judge pattern."""
    
    def __init__(self, llm: Any, prompt_manager: Any):
        """
        Initialize the evaluator with an LLM instance and prompt manager.
        
        Args:
            llm: Initialized LLM instance (e.g., OpenAI, Anthropic, etc.)
            prompt_manager: Prompt management system for constructing evaluation prompts
        """
        self.llm = llm
        self.prompt_manager = prompt_manager
    
    @abstractmethod
    def pre_process(self, answer: str, **kwargs) -> Any:
        """
        Prepare and format the evaluation input.
        
        Args:
            answer: Generated answer to evaluate
            kwargs: Additional template parameters (question, context or golded_answer)
            
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
    def post_process(self, llm_response: str) -> Dict[str, float]:
        """
        Convert LLM response into evaluation scores.
        
        Args:
            llm_response: Raw response string from LLM
            
        Returns:
            Dictionary of evaluation metrics and scores
        """
        pass
    
    def evaluate(self, answer: str, **kwargs) -> Dict[str, float]:
        """
        Main evaluation workflow.
        
        Args:
            answer: Generated answer to evaluate
            kwargs: Additional template parameters (question, context or golded_answer)
            
        Returns:
            Dictionary of evaluation metrics and scores
        """
        processed_data = self.pre_process(answer, **kwargs)
        llm_response = self.call_llm(processed_data)
        return self.post_process(llm_response)