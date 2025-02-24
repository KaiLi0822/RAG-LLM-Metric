from __future__ import annotations
from enum import Enum

from utils.base import BasePrompt


class AnnotationType(BasePrompt):
    KEY_POINT_EXTRACTION = {
        "template": """You are an expert at extracting and generating keywords. 
In this task, you will be given a question, a standard answer and some examples for reference.
Criteria: {criteria}
Examples: {examples}
Test Case:\nQuestion: {question}\nStandard Answer: {golden_answer}\n
{formatter}""",
        "criteria": """Based on the standard answer, you need to summarize the key points necessary to answer the question. 
List them as follows:\n1. …\n2. …\nand so on, as needed.\n\n""",
        "examples": """\nQuestion: What are the significant changes in the newly amended Company Law?\n
Standard Answer: The 2023 amendment to the Company Law introduced several significant changes. 
Firstly, the amendment strengthens the regulation of corporate governance, specifically detailing the responsibilities of the board of directors and the supervisory board[1]. 
Secondly, it introduces mandatory disclosure requirements for Environmental, Social, and Governance (ESG) reports[2]. 
Additionally, the amendment adjusts the corporate capital system, lowering the minimum registered capital requirements[3]. 
Finally, the amendment introduces special support measures for small and medium-sized enterprises to promote their development[4].
\nKey Points:\n1. The amendment strengthens the regulation of corporate governance, detailing the responsibilities of the board of directors and the supervisory board.\n2. 
It introduces mandatory disclosure requirements for ESG reports.\n3. It adjusts the corporate capital system, lowering the minimum registered capital requirements.\n4. 
It introduces special support measures for small and medium-sized enterprises.\n\n
Question: Comparing the major asset acquisitions of Huaxia Entertainment Co., Ltd. in 2017 and Top Shopping Mall in 2018, which company's acquisition amount was larger?\n
Standard Answer: Huaxia Entertainment Co., Ltd.'s asset acquisition amount in 2017 was larger[1], amounting to 120 million yuan[2], whereas Top Shopping Mall's asset acquisition amount in 2018 was 50 million yuan[3].\n
Key Points:\n1. Huaxia Entertainment Co., Ltd.'s asset acquisition amount in 2017 was larger.\n
2. Huaxia Entertainment Co., Ltd.'s asset acquisition amount was 120 million yuan in 2017.\n
3. Top Shopping Mall's asset acquisition amount was 50 million yuan in 2018.\n\n
Question: Comparing the timing of sustainability and social responsibility initiatives by Meihome Housekeeping Services Co., Ltd. and Cultural Media Co., Ltd., which company initiated these efforts earlier?\n
Standard Answer: Meihome Housekeeping Services Co., Ltd. initiated its sustainability and social responsibility efforts earlier[1], in December 2018[2], whereas Cultural Media Co., Ltd. initiated its efforts in December 2019[3].\n
Key Points:\n1. Meihome Housekeeping Services Co., Ltd. initiated its sustainability and social responsibility efforts earlier.\n
2. Meihome Housekeeping Services Co., Ltd. initiated its efforts in December 2018.\n
3. Cultural Media Co., Ltd. initiated its efforts in December 2019.\n\n
Question: Based on the 2017 Environmental and Social Responsibility Report of Green Source Environmental Protection Co., Ltd., how did the company improve community relations through participation in charitable activities, community support and development projects, and public service projects?\n
Standard Answer: Green Source Environmental Protection Co., Ltd. improved community relations through several social responsibility activities.
Firstly, in March 2017, the company participated in or funded charitable activities and institutions to support education, health, and poverty alleviation, enhancing the company's social image and brand recognition[1]. 
Secondly, in June 2017, the company invested in the local community, supporting education, health, and social development projects, deepening its connection with the community and promoting overall community well-being and development[2].
Finally, in August 2017, the company participated in public service projects such as urban greening and public health improvement projects, enhancing the quality of life in the community and promoting sustainable development[3].
These measures enhanced public perception of the company and improved community relations[4].\n
Key Points:\n1. In March 2017, the company participated in or funded charitable activities and institutions to support education, health, and poverty alleviation, enhancing the company's social image and brand recognition.\n
2. In June 2017, the company invested in the local community, supporting education, health, and social development projects, deepening its connection with the community and promoting overall community well-being and development.\n
3. In August 2017, the company participated in public service projects such as urban greening and public health improvement projects, enhancing the quality of life in the community and promoting sustainable development.\n
4. These measures enhanced public perception of the company and improved community relations.\n\n""",
        "formatter": """Respond ONLY with a JSON object containing:\n
- key_points (list of string)
Example:\n
```json\n
'{"key_points": ["1. In March 2017, the company participated in or funded charitable activities and institutions to support education, health, and poverty alleviation, enhancing the company's social image and brand recognition.", "2. In June 2017, the company invested in the local community, supporting education, health, and social development projects, deepening its connection with the community and promoting overall community well-being and development.", "3. In August 2017, the company participated in public service projects such as urban greening and public health improvement projects, enhancing the quality of life in the community and promoting sustainable development.", "4. These measures enhanced public perception of the company and improved community relations."]}'
``` """,
    }

    TRUE_FALSE = {
        "template": """Your task is to follow the criteria described and provide an answer with given question answer and context
        Criteria: {criteria}
        Answer: {answer}
        Context: {context}
        {formatter}""",
        "criteria": """base on the context, judge if the given answer is true or false""",
        "formatter": """Respond ONLY with a JSON object containing:\n
- TF (string of 'true' or 'false')
"Example:\n"
"```json\n"
'{"TF": "TRUE"}'
```" """,
    }

    HAS_NUMERIC_INFO = {
        "template": (
            "Your task is to follow the criteria described then give the answer based on given ground truth answer of a question."
            "Question:{question}\nGround Truth Answer:{golden_answer}\n Criteria: {criteria}\n\n{formatter}"
            ),
        "criteria": ("Analyze the provided question and its corresponding ground truth answer. "
                     "Determine whether the answer critically relies on numerical information, quantifiers, or number-related elements (e.g., ordinals, cardinals, percentages, dates, measurements, counts, ranges, or mathematical values). "
                     "Focus on whether the answer would be incomplete, ambiguous, or incorrect if numerical components were removed."
                     ),
        "formatter": """Respond ONLY with a JSON object containing:\n
- has_numeric_info (string of 'true' or 'false')
"Example:\n"
"```json\n"
'{"has_numeric_info": "true"}'
```" """,
    }

    MISTAKE_GENERATION = {
        "template": """Your task is to generate two versions of an answer based on the ground truth:
        - A perfect paraphrase maintaining all details
        - An incorrect version with specific errors
    
        Ground Truth Answer: {golden_answer}
        Context: {context}
    
        Follow these criteria:
        {criteria_result}
    
        {formatter}""",

        "criteria": lambda mistakes: (
            "First, create a PERFECT PARAPHRASE that:\n"
            "- Preserves all information exactly\n"
            "- Changes only wording/structure\n\n"
            "Then create an INCORRECT VERSION that:\n"
            f"{mistakes}\n"
            "- Clearly shows the specified error types\n"
            "- Maintains grammatical correctness\n"
            "- Do not put multiple mistake into one sentence\n"
        ),

        "formatter": """Respond STRICTLY with JSON containing:
    - "Paraphrased" (exact-meaning version)
    - "Incorrect" (error-containing version) 
    - "Error_Locations" (sentence numbers as list)
    
    Example:
    ```json
    {
        "Paraphrased": "The cardiac cycle consists of systole and diastole phases...",
        "Incorrect": "The cardiac cycle contains systolic and diastolic phases...",
        "Error_Locations": [1, 3]
    }
    ```"""
    }


class AnnotatePromptManager:
    """Manages prompt construction with JSON output formatting"""

    def __init__(self, default_type: AnnotationType = AnnotationType.TRUE_FALSE):
        self.default_type = default_type

    def build_prompt(
            self,
            answer: str = None,
            question: str = None,
            context: str = None,
            eval_type: AnnotationType = None,
            **kwargs
    ) -> str:
        """
        Construct an evaluation prompt with JSON formatting instructions

        Args:
            question: User question/query
            context: Retrieved context used for generation
            answer: Generated answer to evaluate
            eval_type: Type of evaluation to perform
            kwargs: Additional template parameters

        Returns:
            Formatted evaluation prompt with JSON instructions
        """
        eval_type = eval_type or self.default_type

        return eval_type.template.format(
            question=question,
            context=context,
            answer=answer,
            examples=eval_type.examples,
            criteria=eval_type.criteria,
            formatter=eval_type.formatter,
            **kwargs
        )
