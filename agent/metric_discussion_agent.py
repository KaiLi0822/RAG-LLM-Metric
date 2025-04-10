import importlib
import json
import random
from datetime import datetime

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
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
from duckduckgo_search import DDGS

from execution_pipeline.execution_pipeline import CompoundScoreExecutionPipeline
from utils.llm import LLMClient, OpenAIClientLLM


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


def make_valid_identifier(input_str):
    cleaned_str = re.sub(r'[^a-zA-Z0-9_]', '', input_str)
    if cleaned_str and cleaned_str[0].isdigit():
        cleaned_str = '_' + cleaned_str
    return cleaned_str if cleaned_str else 'identifier'


def perform_web_search(query: str) -> str:
    """Perform web search using DuckDuckGo and return results as text."""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        return "\n\n".join([f"Title: {r['title']}\nContent: {r['body']}" for r in results])


class DynamicEvaluationOrchestrator:

    def __init__(self,
                 dataset_name: Optional[str] = None,
                 dataset_df: Optional[pd.DataFrame] = None,
                 evaluate_llm_class: type[LLMClient] = OpenAIClientLLM,
                 evaluate_llm_model: str = "gpt-4o-2024-08-06",
                 evaluate_llm_base_url: str = "https://api.openai.com/v1",
                 agent_llm_model: str = "gpt-4o-2024-08-06",
                 upload_to_hub: bool = True,
                 repo_name: Optional[str] = None,
                 max_discussion_round: Optional[int] = 50):
        if dataset_name is None:
            if upload_to_hub and repo_name is None:
                raise ValueError("must offer repo name when uploading result from pandas df to HF")
            self.dataset = dataset_df
        elif dataset_df is None:
            self.dataset = dataset_name
        else:
            raise ValueError("must offer dataset by name to HF or a pandas dataframe")
        self.evaluate_llm_class = evaluate_llm_class
        self.evaluate_llm_model = evaluate_llm_model
        self.evaluate_llm_base_url = evaluate_llm_base_url
        self.agent_llm_model = agent_llm_model
        self.upload_to_hub = upload_to_hub
        if not repo_name:
            self.repo_name = f"{dataset_name}-Evaluated-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{self.evaluate_llm_model}"
        self.max_discussion_round = max_discussion_round

        self.model_client = self._create_model_client()
        self.base_agents = self._initialize_base_roles()
        self.web_search_tool = self._create_web_search_tool()
        self.search_agent = self._create_search_agent()
        self.domain_detector = self._create_domain_detector()
        self.read_data_tool = self._create_read_data_tool()
        self.example_double_checker = self._create_example_double_checker()
        self.group_chat_summarizer = self._create_group_chat_summarizer()
        self.user_proxy = UserProxyAgent(name="UserProxy")
        self.chat_agent = AssistantAgent(name="ChatAgent", system_message="Your are a helpful assistant",
                                         model_client=self.model_client)
        self.metric_info = self._get_metrics_metadata()

    def _create_model_client(self):
        return OpenAIChatCompletionClient(
            model=self.agent_llm_model,
            api_key=os.getenv("AGENT_OPENAI_API_KEY"),
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "llama",
            },
        )

    def get_sample_data(self, ):
        if isinstance(self.dataset, str):
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError("Hugging Face datasets library required: pip install datasets")
            try:
                hf_dataset = load_dataset(self.dataset, split="train")
                # Random sampling with safety check
                dataset_size = len(hf_dataset)
                n_samples = min(2, dataset_size)

                if dataset_size == 0:
                    return []

                # Generate unique random indices
                indices = random.sample(range(dataset_size), n_samples)
                return json.dumps([{"question": hf_dataset[i]["question"],
                                    "context": hf_dataset[i]["documents"],
                                    "golden_answer": hf_dataset[i]["response"]} for i in indices])

            except Exception as e:
                raise ValueError(f"Failed to load dataset: {str(e)}")

        elif isinstance(self.dataset, pd.DataFrame):
            # Handle pandas DataFrame with random sampling
            n_samples = min(2, len(self.dataset))
            return json.dumps(self.dataset.sample(n=n_samples).rename(columns={"documents": "context",
                                                                               "response": "golden_answer"})[
                                  ["question", "context", "golden_answer"]].to_dict(orient='records'))

        else:
            raise TypeError("Input must be HF dataset name (str) or pandas DataFrame")

    def _get_metrics_metadata(self) -> List[Dict]:
        evaluators = get_evaluator_classes()
        return [evaluator_class.description() for evaluator_class in evaluators]

    def _initialize_base_roles(self) -> Dict[str, AssistantAgent]:
        return {
            "quality_guardian": AssistantAgent(
                name="QualityGuardian",
                system_message="""As the Holistic Quality Guardian, ensure comprehensive evaluation coverage across:
                1. Factual Accuracy: Verify claims against authoritative sources
                2. Contextual Relevance: Maintain topic alignment
                3. Error Robustness: Detect both false positives and negatives
                4. Temporal Consistency: Validate information freshness
                5. Source Diversity: Ensure multi-perspective validation

                Key Responsibilities:
                - Analyze metric coverage across quality dimensions
                - Propose weights based on failure mode criticality
                - Balance precision with practical applicability
                - Collaborate with domain experts for context adaptation
                - Maintain audit trails for metric decisions

                Final output must include verification protocols and weight justifications.""",
                model_client=self.model_client,
            ),
            "user_advocate": AssistantAgent(
                name="UserAdvocate",
                system_message="""As the User Experience Architect, optimize for:
                1. Cognitive Accessibility: Readability across literacy levels
                2. Cultural Appropriateness: Localization requirements
                3. Actionable Clarity: Practical usability of outputs
                4. Engagement Quality: Presentation effectiveness
                5. Accessibility Compliance: WCAG standards adherence

                Key Responsibilities:
                - Evaluate metric sensitivity to user experience factors
                - Propose clarity-weighting strategies
                - Balance technical accuracy with comprehension needs
                - Identify simplification opportunities
                - Validate against diverse user personas

                Final proposals must include accessibility impact assessments.""",
                model_client=self.model_client,
            ),
            "systems_analyst": AssistantAgent(
                name="SystemsAnalyst",
                system_message="""As the Systems Integrity Analyst, ensure evaluation of:
                1. Operational Reliability: Error recovery capabilities
                2. Security Posture: Data protection measures
                3. Performance Characteristics: Latency/throughput impacts
                4. Scalability Factors: Load handling capacities
                5. Compliance Footprint: Regulatory requirements

                Key Responsibilities:
                - Map metrics to system reliability dimensions
                - Weight metrics by operational criticality
                - Analyze failure cascade risks
                - Validate against real-world deployment scenarios
                - Balance rigor with computational costs

                Proposals must include failure mode and effects analysis.""",
                model_client=self.model_client,
            )
        }

    def _create_search_agent(self) -> AssistantAgent:
        return AssistantAgent(
            name="WebSearchAgent",
            system_message="You are an expert web researcher. Your task is to perform a web search "
                           "for the given query and prepare a concise summary of the search results. "
                           "Focus on gathering relevant information that will help in domain identification. "
                           "Respond with a JSON object containing the search query and search results.",
            model_client=self.model_client,
            tools=[self.web_search_tool]
        )

    def _create_domain_detector(self) -> AssistantAgent:
        return AssistantAgent(
            name="DomainAnalyst",
            system_message="You are an expert at analyzing web search results to identify domain categories. "
                           "Given the search results from the previous round, carefully extract and categorize "
                           "the domains represented by the organization or query. "
                           "Respond ONLY with a JSON object containing:\n"
                           "- domains (list of str): Precise domain categories\n"
                           "- reasoning (str): Brief explanation of domain identification\n"
                           "Example:\n"
                           "```json\n"
                           '{"domains": ["technology", "artificial intelligence"], '
                           '"reasoning": "Company focuses on AI research and software development"}'
                           "```",
            model_client=self.model_client
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
            system_message="""You are a helpful Critic for the discussion. Solve tasks using your tools. Your task is 
            to retrieve examples from the evaluation dataset (at least once during the group chat), analyze why the 
            "golden answer" in each example is effective, and validate whether the previously proposed evaluation 
            metrics/importance weights are suitable and make your decision.
            
            If you think the agreement on metrics selection and importance in the discussion has beem made, 
            you should output 'TERMINATE DISCUSSION' Otherwise you should output a JSON with the 
            following format to propose your evaluation metrics selections and weights. {{ "evaluators": [ {{
            "evaluator": "ExactClassName", "weight": 0.25}}, ... ], "rationale": "short explanation" }}""",
            tools=[self.read_data_tool],
            model_client=self.model_client
        )

    def _create_read_data_tool(self) -> FunctionTool:
        return FunctionTool(
            self.get_sample_data, description="retrieve sample data points from evaluate dataset"
        )

    def _create_web_search_tool(self):
        return FunctionTool(
            perform_web_search,
            description="Perform web searches to gather domain-specific information"
        )

    async def detect_domains(self, criteria: str) -> List[str]:
        # First Round: Web Search
        search_message = (f"Perform a comprehensive web search about the mentioned organization/company name if user "
                          f"mentioned in requirements: {criteria}")
        cancellation_token = CancellationToken()
        search_response = await self.search_agent.on_messages(
            [TextMessage(content=search_message, source="system")],
            cancellation_token
        )

        # Extract search results
        try:
            # Try to parse the last message as JSON
            search_results = search_response.chat_message.content
        except (json.JSONDecodeError, IndexError):
            # Fallback to web search if parsing fails
            search_results = "No search results"

        # Second Round: Domain Analysis
        analysis_response = await self.domain_detector.on_messages(
            [TextMessage(content=f"Identify domains in: {criteria}, search_results: {search_results}",
                         source="system")],
            cancellation_token,
        )

        # Extract domain analysis
        try:
            # Try to parse the last message as JSON
            domain_analysis = json.loads(
                analysis_response.chat_message.content.strip().replace("```json", "").replace("```", ""))
        except (json.JSONDecodeError, IndexError, KeyError):
            # Fallback to empty result
            domain_analysis = {
                "domains": [],
                "reasoning": "Unable to determine domains from search results"
            }

        # Combine search results and domain analysis
        return domain_analysis["domains"]

    async def _generate_domain_expert_persona(self, domain: str, user_criteria: str) -> str:
        """Generate system message for domain expert using web search results."""

        search_message = f"Perform a comprehensive web search about{domain} domain best practices for {user_criteria}"
        cancellation_token = CancellationToken()
        search_response = await self.search_agent.on_messages(
            [TextMessage(content=search_message, source="system")],
            cancellation_token
        )
        search_results = search_response.chat_message.content

        prompt = f"""Create a domain expert persona for {domain} that focus on user Requirements and reference with 
        search results:

        Search Results: {search_results}

        User Requirements: {user_criteria}

        The persona should focus on:
        1. Key domain-specific evaluation criteria
        2. Relevant industry standards
        3. Weighting priorities for metrics
        """

        response = await self.chat_agent.on_messages(
            [TextMessage(content=prompt, source="system")],
            cancellation_token
        )
        return response.chat_message.content

    async def select_domain_agents(self, domains: List[str], user_criteria: str) -> List[AssistantAgent]:
        """Dynamically create domain experts with web-powered personas."""
        agents = []
        for domain in domains:
            system_message = await self._generate_domain_expert_persona(domain, user_criteria)
            agents.append(AssistantAgent(
                name=f"{make_valid_identifier(domain.capitalize())}Expert",
                system_message=system_message,
                model_client=self.model_client
            ))
        return agents

    async def negotiate_metrics(self, user_criteria: str) -> Dict:
        domains = await self.detect_domains(user_criteria)
        domain_agents = await self.select_domain_agents(domains, user_criteria)
        all_agents = list(self.base_agents.values()) + domain_agents + [self.example_double_checker]

        evaluator_list = "\n".join(
            [f"- {e['name']}: {e['description']}"
             for e in self.metric_info]
        )

        termination = MaxMessageTermination(self.max_discussion_round) | TextMentionTermination("TERMINATE DISCUSSION")
        group_chat = RoundRobinGroupChat(
            participants=all_agents,
            termination_condition=termination,
        )

        task = f"""User Criteria: {user_criteria}
        Available Evaluators (CLASS NAME : DESCRIPTION):
        {evaluator_list}

        Your group MUST agree on:
        1. Which evaluator classes to use from the available list (NO MORE THAN 5)
        2. Appropriate weights for each (summing to 1.0)
        
        If the User Criteria is specific about the evaluator and weight: 
        1. Check if user criteria is complete (if 
        user-given weights sum up to 1 then it is complete) 
        2. Complete the rest part of the evaluator if user 
        criteria is not complete and keep user provided evaluator and weights unchanged
        
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
        transcripts = "\n".join(
            [msg.content for msg in task_result.messages if isinstance(msg, TextMessage)])
        cancellation_token = CancellationToken()
        response = await self.group_chat_summarizer.on_messages(
            [TextMessage(
                content=(
                    f"Given the user criteria: {user_criteria}\nSummarize the group chat and extract final "
                    f"decision\nGROUP_CHAT: {transcripts}"
                    """Final output MUST be JSON containing:
        {{
            "evaluators": [
                {{"evaluator": "ExactClassName", "weight": 0.25}},
                ...
            ],
            "rationale": "short explanation"
        }}"""),
                source="system")],
            cancellation_token,
        )
        return response.chat_message.content

    def _parse_final_decision(self, response: str) -> Dict:
        try:
            result_dict = json.loads(response.strip().replace("```json", "").replace("```", ""))
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

    async def evaluate(self, user_criteria: str):
        final_result = await self.negotiate_metrics(user_criteria=user_criteria)
        pipeline = CompoundScoreExecutionPipeline(evaluators_with_weights=final_result["classes"])
        if isinstance(self.dataset, str):
            await pipeline.run_pipeline_with_weight(
                dataset_name=self.dataset,
                upload_to_hub=self.upload_to_hub, llm_class=self.evaluate_llm_class,
                repo_id=self.repo_name,
                model=self.evaluate_llm_model,
                base_url=self.evaluate_llm_base_url, )
        else:
            await pipeline.run_pipeline_with_weight(
                dataset_df=self.dataset,
                upload_to_hub=self.upload_to_hub, llm_class=self.evaluate_llm_class,
                repo_id=self.repo_name,
                model=self.evaluate_llm_model,
                base_url=self.evaluate_llm_base_url, )


async def main():
    evaluator = DynamicEvaluationOrchestrator(dataset_name="RAGEVALUATION-HJKMY/TSBC_100row_mistake_added",
                                              evaluate_llm_model="gpt-4o-mini-2024-07-18",
                                              agent_llm_model="gpt-4o-mini-2024-07-18",
                                              max_discussion_round=20)
    await evaluator.evaluate("Please help build evaluate metrics for chatbot run by technical safety BC(TSBC), "
                             "you need to emphasize on the correctness and completeness")


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()
    asyncio.run(main())
