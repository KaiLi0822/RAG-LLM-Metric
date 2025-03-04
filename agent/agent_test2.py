import importlib
import json

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import CancellationToken
from typing import Dict, List, Tuple
import re
import os
import asyncio

import inspect
from importlib.metadata import version, packages_distributions
from evaluator.base_evaluator import RAGEvaluator


def get_evaluator_classes():
    """Retrieve all implemented evaluators derived from RAGEvaluator."""

    # Get module and class info
    module = importlib.import_module('evaluator.evaluators')
    evaluator_classes = []

    for _, cls in inspect.getmembers(module, inspect.isclass):
        if (issubclass(cls, RAGEvaluator) and
                cls.__module__ == module.__name__ and
                cls.__name__.endswith('Evaluator') and
                cls is not RAGEvaluator):
            evaluator_classes.append(cls)

    return evaluator_classes


class DynamicEvaluationOrchestrator:
    def __init__(self):
        self.model_client = self._create_model_client()
        self.base_agents = self._initialize_base_roles()
        self.domain_specialists = self._initialize_domain_roles()
        self.domain_detector = self._create_domain_detector()
        self.group_chat_summarizer = self._create_group_chat_summarizer()
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
                system_message="Focus on factual accuracy and error prevention.",
                model_client=self.model_client,
            ),
            "clarity": AssistantAgent(
                name="ClarityExpert",
                system_message="Advocate for readability and audience comprehension.",
                model_client=self.model_client,
            ),
        }

    def _initialize_domain_roles(self) -> Dict[str, Dict]:
        return {
            "legal": {
                "agent": AssistantAgent(
                    name="LegalGuardian",
                    system_message="Legal Compliance Specialist",
                    model_client=self.model_client,
                ),
            },
            "medical": {
                "agent": AssistantAgent(
                    name="MedicalAuditor",
                    system_message="Healthcare Accuracy Expert",
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
            system_message="""Summarize a multi-agent group chat and strictly follows the uer instruction and user required output formatting""",
            model_client=self.model_client
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
        all_agents = list(self.base_agents.values()) + domain_agents

        evaluator_list = "\n".join(
            [f"- {e['name']}: {e['description']}" for e in self.evaluator_info]
        )

        termination = MaxMessageTermination(10)
        group_chat = RoundRobinGroupChat(
            participants=all_agents,
            termination_condition=termination,
        )

        stream = group_chat.run_stream(
            task=f"User Criteria: {user_criteria}\nAvailable Evaluators:\n{evaluator_list}"
        )
        task_result = await Console(stream)

        return self._parse_final_decision(await self._summarize_group_chat(task_result, user_criteria))

    async def _summarize_group_chat(self, task_result, user_criteria):
        transcripts = "\n".join([msg.content for msg in task_result.messages])
        cancellation_token = CancellationToken()
        response = await self.group_chat_summarizer.on_messages(
            [TextMessage(
                content=f"given the {user_criteria} summarize the group chat and give a final decision \n GROUP_CHAT: {transcripts}", source="system")],
            cancellation_token,
        )
        return response.chat_message.content

    def _parse_final_decision(self, response: str) -> Dict:
        try:
            # TODO: refine output decode
            result_dict = json.loads(response)
            if validation_errors := self._validate_metrics(
                    result_dict.get("metrics", []), result_dict.get("evaluators", [])
            ):
                return {"error": "Validation failed", "details": validation_errors}

            return {
                "metrics": result_dict.get("metrics", []),
                "evaluators": result_dict.get("evaluators", []),
                "rationale": self._extract_rationale(response),
            }
        except Exception as e:
            return {"error": f"Parsing failed: {str(e)}"}

    def _validate_metrics(
            self, metrics: List[Tuple[str, float]], evaluators: List[str]
    ) -> List[str]:
        errors = []
        total_weight = sum(w for _, w in metrics)
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"Weights sum to {total_weight:.2f}")

        # TODO: metric output format 
        return errors

    def _extract_rationale(self, text: str) -> str:
        return re.sub(r".*Rationale:", "", text, flags=re.DOTALL).strip()


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
