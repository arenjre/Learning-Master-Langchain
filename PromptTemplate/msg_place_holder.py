# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv
# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="google/gemma-4-31B-it",
#     task="text-generation",
# )
# model = ChatHuggingFace(llm=llm)

# # chat template
# chat_template = ChatPromptTemplate([
#     ("system", "You are a helpful customer service representative."),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{query}"), 
# ])
# # load msg history
# chat_history = [ ]
# with open("chat_history.txt", "r") as f:
#     chat_history.extend(f.read().splitlines())

# prompt = chat_template.invoke({"chat_history": chat_history, "query": "What is the status of my refund?"})

# # model invoke
# result = model.invoke(prompt.messages)
# print("Bot:", result.content)

# # create prompt




import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# Set up LLM & Prompt
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it", task="text-generation"
)
model = ChatHuggingFace(llm=llm)

chat_template = ChatPromptTemplate(
    [
        ("system", "You are a helpful customer service representative."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}"),
    ]
)

# 1. Define 3-day cutoff threshold
three_days_ago = datetime.now() - timedelta(days=3)

# 2. Read JSON history & filter by date
filtered_chat_history = []

try:
  with open("chat_history.json", "r") as f:
    history_data = json.load(f)

  for entry in history_data:
    msg_time = datetime.fromisoformat(entry["timestamp"])

    # Keep only messages newer than 3 days
    if msg_time >= three_days_ago:
      if entry["role"] == "human":
        filtered_chat_history.append(HumanMessage(content=entry["content"]))
      elif entry["role"] == "ai":
        filtered_chat_history.append(AIMessage(content=entry["content"]))
except FileNotFoundError:
  pass

# 3. Invoke model with filtered (shorter) prompt
prompt = chat_template.invoke({
    "chat_history": filtered_chat_history,
    "query": "What is the status of my refund?",
})

result = model.invoke(prompt.messages)
print("Bot:", result.content)