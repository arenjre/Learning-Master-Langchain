'''
BaseTool is a abstract class for all tools. It defines the interface that all tools must follow.
'''


from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


# 1. Define the input schema
class CalculatorInput(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")


# 2. Create a custom tool using BaseTool
class MultiplyTool(BaseTool):
    name: str = "multiply_numbers"
    description: str = "Multiply two numbers."
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b


# 3. Create the tool object
multiply_tool = MultiplyTool()


# 4. Invoke the tool
result = multiply_tool.invoke({
    "a": 10,
    "b": 5
})

print(result)
