import asyncio

from agent.metric_discussion_agent import DynamicEvaluationOrchestrator


async def main():
    evaluator = DynamicEvaluationOrchestrator(dataset_name="RAGEVALUATION-HJKMY/TSBC_cleaned_demo",
                                              evaluate_llm_model="gpt-4o-mini-2024-07-18",
                                              agent_llm_model="gpt-4o-mini-2024-07-18",
                                              max_discussion_round=20)
    await evaluator.evaluate("Please help build evaluate metrics for chatbot run by technical safety BC(TSBC)")


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()
    asyncio.run(main())