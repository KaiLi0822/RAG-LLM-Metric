from enum import Enum


class BasePrompt(Enum):
    """Base class for prompt enums with template and output formatting"""

    @property
    def template(self) -> str:
        return self.value['template']

    @property
    def criteria(self) -> str:
        return self.value.get('criteria', '')

    @property
    def formatter(self) -> str:
        return self.value['formatter']

    @property
    def examples(self) -> str:
        return self.value.get('examples', '')

    @classmethod
    def get_prompt_type(cls, name: str) -> 'BasePrompt':
        return cls[name.upper()]

