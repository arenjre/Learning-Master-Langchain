# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv
# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="google/gemma-4-31B-it",  # or "Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"
#     task="text-generation",
# )

# model = ChatHuggingFace(llm=llm)

# chat_history = [
#     SystemMessage(content="You are a helpful assistant.")
# ]

# while True:
#     user_input = input("User: ")
#     if user_input.lower() == "exit":
#         break
#     chat_history.append(HumanMessage(content=user_input))
#     result = model.invoke(chat_history)
#     ai_msg = AIMessage(content=result.content)
#     chat_history.append(ai_msg)
#     print("Bot:", result.content)


# # while True:
# #     user_input = input("User: ")
# #     if user_input.lower() == "exit":
# #         break
# #     result = model.invoke(user_input)
# #     print("Bot:", result.content)


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

system_message = SystemMessage(
    content="You are a helpful assistant."
)

chat_history = []
conversation_summary = ""


while True:

    user_input = input("User: ")

    if user_input.lower() == "exit":
        break

    chat_history.append(
        HumanMessage(content=user_input)
    )

    # Build prompt
    messages = [system_message]

    if conversation_summary:
        messages.append(
            SystemMessage(
                content=f"Conversation summary:\n{conversation_summary}"
            )
        )

    messages.extend(chat_history[-6:])

    # -----------------------------------
    # Count input words
    # -----------------------------------
    prompt_text = "\n".join(
        message.content for message in messages
    )

    word_count = len(prompt_text.split())

    print(f"\nInput prompt word count: {word_count}")

    # -----------------------------------
    # Send prompt
    # -----------------------------------
    result = model.invoke(messages)

    print("Bot:", result.content)

    chat_history.append(
        AIMessage(content=result.content)
    )