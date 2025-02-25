from llm import OpenAIClientLLM
from prompt_manager import PromptManager, EvaluationType
import json


class MetaReviewerLearningFacilitation:
    def __init__(self, llm: OpenAIClientLLM, prompt_manager: PromptManager):
        self.llm = llm
        self.prompt_manager = prompt_manager

    def build_meta_prompt(self, score1, reason1, score2, reason2):
        template = (
            "Evaluate the combined learning facilitation score based on the following inputs:\n"
            "Score 1: {score1}\nReason 1: {reason1}\n"
            "Score 2: {score2}\nReason 2: {reason2}\n"
            "Consider these aspects: {criteria}\n\n"
            "{formatter}"
        )
        criteria = EvaluationType.LEARNING_FACILITATION.criteria
        formatter = EvaluationType.LEARNING_FACILITATION.formatter
        return template.format(
            score1=score1,
            reason1=reason1,
            score2=score2,
            reason2=reason2,
            criteria=criteria,
            formatter=formatter
        )

    def get_meta_scores(self, score1, reason1, score2, reason2):
        prompt = self.build_meta_prompt(score1, reason1, score2, reason2)
        response = self.llm.generate(prompt)
        return self.parse_response(response)

    def parse_response(self, response):
        response_text = response.strip().replace('```json', '').replace('```', '')
        result = json.loads(response_text)
        return {
            "learning_facilitation_score": result['learning_facilitation_score'],
            "educational_strengths": result['educational_strengths'],
            "areas_for_improvement": result['areas_for_improvement'],
            "confidence": result['confidence']
        }


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    llm=OpenAIClientLLM(
        model="gpt-4o",
        base_url='https://api.openai.com/v1/'
    )
    prompt_manager = PromptManager(
        default_type=EvaluationType.LEARNING_FACILITATION)
    meta_reviewer = MetaReviewerLearningFacilitation(llm, prompt_manager)

    score1 = 0.8
    reason1 = "Clear explanations and good examples."
    score2 = 0.7
    reason2 = "Needs more depth and visual aids."

    meta_scores = meta_reviewer.get_meta_scores(
        score1, reason1, score2, reason2)
    print(meta_scores)
