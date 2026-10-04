'''
structured tool are those tools that have a structured input and output. 
The input and output are defined using Pydantic models. 
This allows for better validation and documentation of the tool's input and output.

that follows a structured schema, basically pydantic models
'''

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")

def multiply(a: float, b: float) -> float:
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply,
    name="multiply_numbers",
    description="Multiply two numbers.",
    args_schema=CalculatorInput
)

result = multiply_tool.invoke({
    "a": 10,
    "b": 5
})

print(result)



# discount tool example

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class DiscountInput(BaseModel):
    price: float = Field(gt=0, description="Original product price")
    discount: float = Field(
        ge=0, le=100,
        description="Discount percentage from 0 to 100"
    )

def calculate_discount(price: float, discount: float) -> float:
    return round(price * (1 - discount / 100), 2)

discount_tool = StructuredTool.from_function(
    func=calculate_discount,
    name="calculate_discount",
    description="Calculate final price after discount.",
    args_schema=DiscountInput
)

result = discount_tool.invoke({
    "price": 2000,
    "discount": 15
})

print(result)