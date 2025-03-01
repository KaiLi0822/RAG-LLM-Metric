from utils.llm import OpenAIClientLLM
from evaluator.prompt_manager import EvalPromptManager, EvaluationType
import json

class MetaReviewerBase:
    def __init__(self, llm: OpenAIClientLLM, prompt_manager: EvalPromptManager, evaluation_type: EvaluationType):
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.evaluation_type = evaluation_type

    def build_meta_prompt(self, score1, reason1, score2, reason2):
        template = (
            "Evaluate the combined {evaluation_type} score based on the following inputs:\n"
            "Score 1: {score1}\nReason 1: {reason1}\n"
            "Score 2: {score2}\nReason 2: {reason2}\n"
            "Consider these aspects: {criteria}\n\n"
            "{formatter}"
        )
        criteria = self.evaluation_type.criteria
        formatter = self.evaluation_type.formatter
        return template.format(
            evaluation_type=self.evaluation_type.name,
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
        parsed_result = {}
        for key, value in result.items():
            parsed_result[key] = value
        return parsed_result

def create_meta_reviewer_classes():
    meta_reviewer_classes = {}
    for eval_type in EvaluationType:
        class_name = f"MetaReviewer{eval_type.name.capitalize()}"
        meta_reviewer_classes[eval_type.name] = type(
            class_name,
            (MetaReviewerBase,),
            {"__init__": lambda self, llm, prompt_manager, eval_type=eval_type: MetaReviewerBase.__init__(self, llm, prompt_manager, eval_type)}
        )
    return meta_reviewer_classes

# Create all MetaReviewer classes
meta_reviewer_classes = create_meta_reviewer_classes()

# Example usage
if __name__ == "__main__":
    llm = OpenAIClientLLM()
    prompt_manager = EvalPromptManager()
    MetaReviewerLearningFacilitation = meta_reviewer_classes["LEARNING_FACILITATION"]
    meta_reviewer = MetaReviewerLearningFacilitation(llm, prompt_manager)
    scores = meta_reviewer.get_meta_scores(0.8, "Good explanation", 0.7, "Needs more examples")
    print(scores)