from langchain.tools import tool, InjectedToolArg
import requests
from typing import Annotated
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage
)
from dotenv import load_dotenv
load_dotenv()
import json


EXCHANGE_RATE_API_KEY = "bb75677e0b91603baea68951"

@tool
def get_conversion_rate(
    base_currency: str,
    target_currency: str
) -> float:
    """
    Get the current exchange rate from base_currency
    to target_currency.
    """

    base_currency = base_currency.upper()
    target_currency = target_currency.upper()

    url = (
        f"https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_API_KEY}/pair/{base_currency}/{target_currency}"
    )

    response = requests.get(
        url,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    if data["result"] != "success":
        raise ValueError(
            f"API Error: {data.get('error-type')}"
        )

    # Returns FLOAT
    return data["conversion_rate"]

@tool
def convert_currency(
        amount: float, 
        conversion_rate: Annotated[float, InjectedToolArg]
    ) -> float:
    '''This function converts the amount from base_currency to target_currency using the conversion_rate'''
    return amount * conversion_rate

tools=[get_conversion_rate, convert_currency]

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)
model_with_tools = model.bind_tools(tools=tools)

messages = []

user_message = HumanMessage(
        content=(
            "I want to convert 100 USD to INR. "
            "What is the current exchange rate and converted amount?"
        )
    )
messages.append(user_message)


# ------------------------------------------------
# Call model FIRST time
# ------------------------------------------------
ai_message = model_with_tools.invoke(messages)

# IMPORTANT: append AIMessage directly
messages.append(ai_message)


conversion_rate = None

# ------------------------------------------------
# Tool Execution
# Note: Tool calling and Tool execution are separate steps. The model generates tool calls, and the application/user executes them.
# ------------------------------------------------

for tool_call in ai_message.tool_calls:

    if tool_call["name"] == "get_conversion_rate":

        conversion_rate = get_conversion_rate.invoke(tool_call)

        # Return tool result to LLM
        messages.append(
            ToolMessage(
                content=str(conversion_rate),
                tool_call_id=tool_call["id"]
            )
        )
    elif tool_call["name"] == "convert_currency":
        args = tool_call["args"].copy()
        args["conversion_rate"] = float(conversion_rate.content)
        converted_amount = convert_currency.invoke(args)

        messages.append(
            ToolMessage(
                content=str(converted_amount),
                tool_call_id=tool_call["id"]
            )
        )

final_response = model_with_tools.invoke(messages)
print("Final response\n")
print(final_response.content)