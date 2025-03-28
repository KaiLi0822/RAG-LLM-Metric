import importlib
import json
import sys
sys.path.append("..")

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
        self.read_data_tool = self._create_read_data_tool()
        self.example_double_checker = self._create_example_double_checker()
        self.group_chat_summarizer = self._create_group_chat_summarizer()
        self.user_proxy = UserProxyAgent(name="UserProxy")

    def _create_model_client(self):
        return OpenAIChatCompletionClient(
            model=os.getenv("MODEL_ID", "meta-llama/Llama-3.3-70B-Instruct"),
            base_url=os.getenv("BASE_URL", "https://api-eu.centml.com/openai/v1"),
            api_key=os.getenv("OPENAI_API_KEY"),
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "llama",
            },
        )

    def _initialize_base_roles(self) -> Dict[str, AssistantAgent]:
        json_dir = os.path.join(os.path.dirname(__file__), '../output/json')
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

        base_roles = {}
        for json_file in json_files:
            with open(os.path.join(json_dir, json_file), 'r') as f:
                data = json.load(f)
                first_line = data.get('persona').split('\n')[0]
                role_name = first_line.split(':')[1].strip()
                # Sanitize role_name to be a valid Python identifier
                role_name = re.sub(r'\W|^(?=\d)', '_', role_name)
                system_message = data.get('persona')
                base_roles[role_name] = AssistantAgent(
                    name=role_name,
                    system_message=system_message,
                    model_client=self.model_client,
                )
        return base_roles

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
            get_sample_data, description="retrieve data from user's dataset"
        )

    async def evaluate(self, user_criteria: str) -> Dict:
        # Directly evaluate with the given prompt
        task = f"""User Criteria: {user_criteria}

        Your task is to evaluate the given criteria and provide a score.

        Respond ONLY with a JSON object containing:
        Example:
        ```json\n
        {{
            "score": 0.85,
            "rationale": "short explanation"
        }}
        ```
        """

        # Use only base agents and example double checker
        all_agents = list(self.base_agents.values()) + [self.example_double_checker]

        termination = MaxMessageTermination(max_messages=3)
        group_chat = RoundRobinGroupChat(
            participants=all_agents,
            termination_condition=termination,
        )

        stream = group_chat.run_stream(task=task)
        task_result = await Console(stream)

        return self._parse_final_decision(await self._summarize_group_chat(task_result, user_criteria))

    async def _summarize_group_chat(self, task_result, user_criteria):
        transcripts = "\n".join([msg.content for msg in task_result.messages])
        cancellation_token = CancellationToken()
        response = await self.group_chat_summarizer.on_messages(
            [TextMessage(
                content=f"""Given the user criteria: {user_criteria}
                Summarize the group chat and extract final decision.
                Ensure the output is a valid JSON object in the following format:
                {{
                    "score": 0.85,
                    "rationale": "short explanation"
                }}
                GROUP_CHAT: {transcripts}""",
                source="system")],
            cancellation_token,
        )
        return response.chat_message.content

    def _parse_final_decision(self, response: str) -> Dict:
        try:
            if response.startswith("```json"):
                response = response.replace("```json\n", "", 1).replace("```", "", 1).strip()
            result_dict = json.loads(response)
            score = result_dict.get("score", 0)
            rationale = result_dict.get("rationale", "")

            return {
                "score": score,
                "rationale": rationale
            }
        except Exception as e:
            return {"error": f"Parsing failed: {str(e)}"}

async def main():
    
    evaluator = DynamicEvaluationOrchestrator()

    # Legal example
    legal_result = await evaluator.evaluate(
        "Legal document assistant requiring strict compliance and citation accuracy"
    )
    print("Legal Evaluation Setup:", legal_result)


if __name__ == "__main__":
    import dotenv
    max_messages=3

    dotenv.load_dotenv()
    asyncio.run(main())