import asyncio

from agent.metric_discussion_agent import DynamicEvaluationOrchestrator


async def main():
    evaluator = DynamicEvaluationOrchestrator(
        dataset_name="RAGEVALUATION-HJKMY/TSBC_100row_mistake_added",
        evaluate_llm_model="o1-2024-12-17",
        agent_llm_model="gpt-4o-mini-2024-07-18",
        max_discussion_round=20,
    )
    await evaluator.evaluate(
        "Please help build evaluate metrics for chatbot run by technical safety BC(TSBC), "
        "Here are the metrics and their weights"
        """- FactualAccuracyEvaluator: 20%
- FactualCorrectnessEvaluator: 15%
- KeyPointCompletenessEvaluator: 20%
- KeyPointHallucinationEvaluator: 15%
- ContextRelevanceEvaluator: 10%
- CoherenceEvaluator: 10%
- EngagementEvaluator: 10%"""
    )


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()
    asyncio.run(main())
