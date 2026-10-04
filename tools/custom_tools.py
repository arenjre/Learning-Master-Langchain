# 1. using tool decorator to create a custom tool
from langchain.tools import tool

@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b


result = multiply_numbers.invoke({"a": 3,"b": 4})
print(multiply_numbers.args)  # Output: {'a': 3, 'b': 4}
print(result)  # Output: 12

