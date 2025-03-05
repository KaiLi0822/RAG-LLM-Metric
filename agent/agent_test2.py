import importlib
import json

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import CancellationToken
from typing import Dict, List, Optional, Tuple
import re
import os
import asyncio

import inspect

import pandas as pd
from evaluator.base_evaluator import RAGEvaluator


def get_evaluator_classes():
    """Retrieve all implemented evaluators derived from RAGEvaluator."""
    module = importlib.import_module('evaluator.evaluators')
    evaluator_classes = []

    for _, cls in inspect.getmembers(module, inspect.isclass):
        if (issubclass(cls, RAGEvaluator) and
                cls.__module__ == module.__name__ and
                cls.__name__.endswith('Evaluator') and
                cls is not RAGEvaluator):
            evaluator_classes.append(cls)

    return evaluator_classes


def get_sample_data():
    # TODO:
    raise NotImplementedError


class DynamicEvaluationOrchestrator:

    def __init__(self,
                 dataset_name: Optional[str] = None,
                 dataset_df: Optional[pd.DataFrame] = None, ):
        if dataset_name is None:
            self.dataset = []
        elif dataset_df is None:
            self.dataset = []
        else:
            raise ValueError("must offer dataset by name to HF or a pandas dataframe")
        self.model_client = self._create_model_client()
        self.base_agents = self._initialize_base_roles()
        self.domain_specialists = self._initialize_domain_roles()
        self.domain_detector = self._create_domain_detector()
        self.example_double_checker = self._create_example_double_checker()
        self.group_chat_summarizer = self._create_group_chat_summarizer()
        self.read_data_tool = self._create_read_data_tool()
        self.user_proxy = UserProxyAgent(name="UserProxy")
        self.evaluator_info = self._get_evaluator_metadata()
        self.metric_map = self._create_metric_map()

    def _create_model_client(self):
        return OpenAIChatCompletionClient(
            model="meta-llama/Llama-3.3-70B-Instruct",
            base_url="https://api-eu.centml.com/openai/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "llama",
            },
        )

    def _get_evaluator_metadata(self) -> List[Dict]:
        evaluators = get_evaluator_classes()
        return [evaluator_class.description() for evaluator_class in evaluators]

    def _create_metric_map(self) -> Dict[str, List[str]]:
        return {
            "accuracy": ["evaluate_factual_correctness"],
            "relevance": ["evaluate_context_relevance"],
            # Add other metric mappings
        }

    def _initialize_base_roles(self) -> Dict[str, AssistantAgent]:
        return {
            "precision": AssistantAgent(
                name="PrecisionExpert",
                system_message="""Focus on factual accuracy and error prevention. Discuss RAGEvaluator classes and their weights.
                Consider appropriate weights for accuracy-related evaluators. Final output must include class names and weights.""",
                model_client=self.model_client,
            ),
            "clarity": AssistantAgent(
                name="ClarityExpert",
                system_message="""Advocate for readability and comprehension. Negotiate weights for clarity-related evaluators.
                Ensure final list includes tuple of (EvaluatorClassName, weight) for clarity metrics.""",
                model_client=self.model_client,
            ),
        }

    def _initialize_domain_roles(self) -> Dict[str, Dict]:
        return {
            "legal": {
                "agent": AssistantAgent(
                    name="LegalGuardian",
                    system_message="""Legal compliance specialist. Propose (ClassName, weight) tuples for legal evaluators.
                    Defend weights using domain requirements. Collaborate on final list structure.""",
                    model_client=self.model_client,
                ),
            },
            "medical": {
                "agent": AssistantAgent(
                    name="MedicalAuditor",
                    system_message="""Healthcare accuracy expert. Argue for medical evaluator classes and weights.
                    Ensure final output contains proper (ClassName, weight) tuples.""",
                    model_client=self.model_client,
                ),
            },
        }

    def _create_domain_detector(self) -> AssistantAgent:
        return AssistantAgent(
            name="DomainAnalyst",
            system_message="Analyze requirements and respond with domains: legal, medical, technical",
            model_client=self.model_client,
        )

    def _create_group_chat_summarizer(self) -> AssistantAgent:
        return AssistantAgent(
            name="GroupChatSummarizer",
            system_message="""Extract final agreed list of (EvaluatorClassName, weight) tuples from discussion.
            Format STRICTLY as: {
                "evaluators": [
                    {"evaluator": "ExactClassName", "weight": 0.25},
                    ...
                ],
                "rationale": "summary"
            }""",
            model_client=self.model_client
        )

    def _create_example_double_checker(self) -> AssistantAgent:
        return AssistantAgent(
            name="ExampleDoubleChecker",
            system_message="""You are a helpful AI assistant. Solve tasks using your tools. Your task is to retrieve 
            examples from the evaluation dataset, analyze why the "golden answer" in each example is effective, 
            and validate whether the previously proposed evaluation metrics/importance weights are suitable.""",
            tools=[self.read_data_tool],
            model_client=self.model_client
        )

    def _create_read_data_tool(self) -> FunctionTool:
         return FunctionTool(
            get_sample_data, description="reterive data from user's dataset"
        )

    async def detect_domains(self, criteria: str) -> List[str]:
        cancellation_token = CancellationToken()
        response = await self.domain_detector.on_messages(
            [TextMessage(content=f"Identify domains in: {criteria}", source="system")],
            cancellation_token,
        )
        return list(
            set(
                re.findall(
                    r"legal|medical|technical", response.chat_message.content.lower()
                )
            )
        )

    def select_domain_agents(self, domains: List[str]) -> List[AssistantAgent]:
        return [
            self.domain_specialists[d]["agent"]
            for d in domains
            if d in self.domain_specialists
        ]

    async def negotiate_metrics(self, user_criteria: str) -> Dict:
        domains = await self.detect_domains(user_criteria)
        domain_agents = self.select_domain_agents(domains)
        all_agents = list(self.base_agents.values()) + domain_agents + [self.example_double_checker]

        evaluator_list = "\n".join(
            [f"- {e['name']} ({e['class_name']}): {e['description']}"
             for e in self.evaluator_info]
        )

        termination = MaxMessageTermination(10)
        group_chat = RoundRobinGroupChat(
            participants=all_agents,
            termination_condition=termination,
        )

        task = f"""User Criteria: {user_criteria}
        Available Evaluators (CLASS NAME : DESCRIPTION):
        {evaluator_list}

        Your group MUST agree on:
        1. Which evaluator classes to use from the available list
        2. Appropriate weights for each (summing to 1.0)

        Final output MUST be JSON containing:
        {{
            "evaluators": [
                {{"evaluator": "ExactClassName", "weight": 0.25}},
                ...
            ],
            "rationale": "short explanation"
        }}"""

        stream = group_chat.run_stream(task=task)
        task_result = await Console(stream)

        return self._parse_final_decision(await self._summarize_group_chat(task_result, user_criteria))

    async def _summarize_group_chat(self, task_result, user_criteria):
        transcripts = "\n".join([msg.content for msg in task_result.messages])
        cancellation_token = CancellationToken()
        response = await self.group_chat_summarizer.on_messages(
            [TextMessage(
                content=f"Given the user criteria: {user_criteria}\nSummarize the group chat and extract final decision\nGROUP_CHAT: {transcripts}",
                source="system")],
            cancellation_token,
        )
        return response.chat_message.content

    def _parse_final_decision(self, response: str) -> Dict:
        try:
            # TODO: refine output decode
            result_dict = json.loads(response)
            evaluator_data = result_dict.get("evaluators", [])

            evaluator_classes = {cls.__name__: cls for cls in get_evaluator_classes()}
            evaluator_tuples = []

            for item in evaluator_data:
                cls_name = item.get("evaluator")
                weight = item.get("weight")
                if cls := evaluator_classes.get(cls_name):
                    evaluator_tuples.append((cls, float(weight)))

            if validation_errors := self._validate_metrics(evaluator_tuples):
                return {"error": "Validation failed", "details": validation_errors}

            self.process_final_decision(evaluator_tuples)

            return {
                "evaluators": [(cls.__name__, weight) for cls, weight in evaluator_tuples],
                "rationale": self._extract_rationale(response),
                "classes": evaluator_tuples
            }
        except Exception as e:
            return {"error": f"Parsing failed: {str(e)}"}

    def _validate_metrics(self, evaluators: List[Tuple[RAGEvaluator, float]]) -> List[str]:
        errors = []
        total_weight = sum(w for _, w in evaluators)
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"Invalid weights sum: {total_weight:.2f} (must sum to 1.0)")

        for cls, weight in evaluators:
            if not (0 <= weight <= 1):
                errors.append(f"Invalid weight {weight:.2f} for {cls.__name__}")

        return errors

    def _extract_rationale(self, text: str) -> str:
        return re.sub(r".*Rationale:", "", text, flags=re.DOTALL).strip()

    def process_final_decision(self, evaluators: List[Tuple[RAGEvaluator, float]]):
        """Example function to process the final decision"""
        print("\n=== FINAL EVALUATION PLAN ===")
        for evaluator_cls, weight in evaluators:
            print(f"- {evaluator_cls.__name__}: {weight:.0%}")
        print("=== END OF PLAN ===\n")
        return evaluators


async def main():
    evaluator = DynamicEvaluationOrchestrator()

    # Legal example
    legal_result = await evaluator.negotiate_metrics(
        "Legal document assistant requiring strict compliance and citation accuracy"
    )
    print("Legal Evaluation Setup:", legal_result)

    # Medical example
    medical_result = await evaluator.negotiate_metrics(
        "Clinical decision support system needing medical accuracy"
    )
    print("\nMedical Evaluation Setup:", medical_result)


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()
    asyncio.run(main())
