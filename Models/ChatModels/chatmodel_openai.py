from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(model_name="gpt-4.1-mini",)

result = model.invoke("What is the capital of France?")

print(result.content)

