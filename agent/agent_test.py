from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, ChatResult
from typing import Dict, List, Tuple
import re
import os


class DynamicEvaluationOrchestrator:
    def __init__(self):
        self.base_agents = self._initialize_base_roles()
        self.domain_specialists = self._initialize_domain_roles()
        self.domain_detector = self._create_domain_detector()
        self.user_proxy = UserProxyAgent(name="UserProxy", human_input_mode="NEVER")

    def _llm_config(self) -> Dict:
        return {"config_list": [
            {"model": "meta-llama/Llama-3.3-70B-Instruct",
             "base_url": "https://api-eu.centml.com/openai/v1",
             "api_key": os.getenv("OPENAI_API_KEY")}]}

    def _initialize_base_roles(self) -> Dict[str, AssistantAgent]:
        return {
            "precision": AssistantAgent(
                name="PrecisionExpert",
                system_message="Focus on factual accuracy and error prevention. Push for strict verification metrics.",
                llm_config=self._llm_config()
            ),
            "clarity": AssistantAgent(
                name="ClarityExpert",
                system_message="Advocate for readability and audience comprehension metrics.",
                llm_config=self._llm_config()
            ),
        }

    def _initialize_domain_roles(self) -> Dict[str, Dict]:
        return {
            "legal": {
                "agent": AssistantAgent(
                    name="LegalGuardian",
                    system_message="""Legal Compliance Specialist:
                    - Prioritize regulatory adherence
                    - Require citation verification
                    - Strict accuracy requirements""",
                    llm_config=self._llm_config()
                ),
                "triggers": ["legal", "compliance", "regulation"]
            },
            "medical": {
                "agent": AssistantAgent(
                    name="MedicalAuditor",
                    system_message="""Healthcare Accuracy Expert:
                    - Focus on clinical validity
                    - Require medical source verification
                    - Patient safety metrics""",
                    llm_config=self._llm_config()
                ),
                "triggers": ["medical", "healthcare", "clinical"]
            },
            "technical": {
                "agent": AssistantAgent(
                    name="TechnicalValidator",
                    system_message="""Technical Documentation Specialist:
                    - Verify technical specifications
                    - Check code/equation accuracy
                    - Validate implementation steps""",
                    llm_config=self._llm_config()
                ),
                "triggers": ["technical", "engineering", "specification"]
            }
        }

    def _create_domain_detector(self) -> AssistantAgent:
        return AssistantAgent(
            name="DomainAnalyst",
            system_message="""Analyze user requirements to identify domains. 
            Respond ONLY with comma-separated domain keywords from: legal, medical, technical""",
            llm_config=self._llm_config()
        )

    def detect_domains(self, criteria: str) -> List[str]:
        self.user_proxy.initiate_chat(
            self.domain_detector,
            message=f"Identify domains in: {criteria}",
            clear_history=True,
            max_turns=1,
        )
        response = self.domain_detector.last_message()["content"]
        return list(set(re.findall(r"legal|medical|technical", response.lower())))

    def select_domain_agents(self, domains: List[str]) -> List[AssistantAgent]:
        selected = []
        for domain in domains:
            if specialist := self.domain_specialists.get(domain):
                selected.append(specialist["agent"])
        return selected[:2]

    def negotiate_metrics(self, user_criteria: str) -> Dict:
        domains = self.detect_domains(user_criteria)
        print(f"Detected domains: {domains}")

        domain_agents = self.select_domain_agents(domains)
        all_agents = list(self.base_agents.values()) + domain_agents + [self.user_proxy]

        group_chat = GroupChat(
            agents=all_agents,
            messages=[],
            max_round=6,
            speaker_selection_method="round_robin"
        )

        manager = GroupChatManager(
            groupchat=group_chat,
            system_message="""Facilitate metric selection considering:
            1. Domain-specific requirements
            2. Base quality dimensions
            3. Weight distribution
            Final output must be a Python list of tuples with metric names and weights""",
            llm_config=self._llm_config()
        )

        group_chat_result = self.user_proxy.initiate_chat(
            manager,
            message=f"User Criteria: {user_criteria}\n\nGenerate weighted metrics considering all perspectives.",
            max_turns=1
        )

        return self._parse_final_decision(group_chat_result)

    def _parse_final_decision(self, group_chat_result: ChatResult) -> Dict:
        final_message = group_chat_result.chat_history[0]["content"]
        try:
            metrics = eval(re.search(r"\[.*\]", final_message, re.DOTALL).group())
            return {"metrics": metrics, "rationale": self._extract_rationale(final_message)}
        except:
            return {"error": "Failed to parse agent response"}

    def _extract_rationale(self, text: str) -> str:
        return re.sub(r".*Rationale:", "", text, flags=re.DOTALL).strip()


# Usage Example
async def main():
    evaluator = DynamicEvaluationOrchestrator()

    legal_result = evaluator.negotiate_metrics(
        "Legal document assistant requiring strict compliance with regulations "
        "and perfect citation accuracy, must maintain readability for non-experts"
    )
    print("Legal Results:")
    print(legal_result)

    medical_result = evaluator.negotiate_metrics(
        "Clinical decision support system needing medical accuracy and "
        "technical implementation validation, must prioritize patient safety"
    )
    print("\nMedical Results:")
    print(medical_result)


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()
    import asyncio

    asyncio.run(main())
