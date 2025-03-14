from pydantic import BaseModel, Field


class PromptStrategy(BaseModel):
    class_strategies: dict[str, str] = Field(
        ...,
        description="A strategy and guidelines for generating prompts that will be used for generating synthetic images. Class_name: strategy",
    )

    def __str__(self):
        return f"{self.__class__.__name__}: {self.model_dump_json(indent=2)}"

    def __getitem__(self, key):
        # Return the strategy if it exists, otherwise return a default value
        return self.class_strategies.get(
            key,
            "No strategy for now. Generate prompts randomly based on the class name.",
        )


class ClassDetails(BaseModel):
    performance: float = Field(..., description="Performance of the class")
    prompt_summary: str = Field(..., description="Summary of the prompts")


class ClassInformation(BaseModel):
    class_details: dict[str, ClassDetails] = Field(
        ..., description="Details of all classes"
    )

    def __str__(self):
        return f"{self.__class__.__name__}: {self.model_dump_json(indent=2)}"


class PromptSummaries(BaseModel):
    class_prompt_summaries: dict[str, str] = Field(
        ..., description="Summaries of all prompts in the format class_name: summary"
    )  #


class Prompts(BaseModel):
    prompts: list[str] = Field(..., description="List of prompts")
