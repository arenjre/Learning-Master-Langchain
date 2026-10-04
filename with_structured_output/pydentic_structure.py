from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field, EmailStr
from typing import Literal, Optional

from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI()

_review = (
    "Hi, My name is Anand Kumar(someemail@gmail.com), Despite solid build quality and snappy app switching with 12GB RAM, "
    "the thermal throttling under high gaming loads and inconsistent fast-charging speeds hold this device back, so based on this i would give 3.8 start"

)

# 1. Define the schema using Field for descriptions
class Review(BaseModel):
    summary: str = Field(description="A brief summary of the review, in 1-2 sentences")
    key_themes: list[str] = Field(description="List of key themes discussed in the review")
    cons: list[str] = Field(description="List of negative aspects mentioned in the review")
    pros: list[str] = Field(description="List of positive aspects mentioned in the review")
    sentiment: Literal['Positive', 'Negative', 'Neutral'] = Field(
        description="The overall sentiment of the review"
    )
    name: Optional[str] = Field(description="The name of the reviewer, if available")
    reviewer_email: Optional[str] = Field(description="Email of the reviewer, if available")
    review_start: float = Field(gt=0, lt=6)

# 2. Set up LLM & Parser
# llm = HuggingFaceEndpoint(
#     repo_id="google/gemma-4-31B-it",
#     task="text-generation",
# )
# model = ChatHuggingFace(llm=llm)

# Execute
structured_output = model.with_structured_output(Review)
result = structured_output.invoke(_review)
print(result)
