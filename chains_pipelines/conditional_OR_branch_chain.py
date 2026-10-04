from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableBranch

from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation",
)

billing_prompt = PromptTemplate(
    template = "You are a billing expert. Answer this billing question: {question}",
    input_variables=["question"]
)

technical_prompt = PromptTemplate(
    template = "You are a technical support expert. Answer this question: {question}",
    input_variables=["question"]
)

general_prompt = PromptTemplate(
    template = "You are a customer support agent. Answer this question: {question}",
    input_variables=["question"]
)

parser = StrOutputParser()
model = ChatHuggingFace(llm=llm)

billing_chain = billing_prompt | model | parser
technical_chain = technical_prompt | model | parser
general_chain = general_prompt | model | parser

branch_chain = RunnableBranch(

    (
        lambda x: x["category"] == "billing",
        billing_chain
    ),

    (
        lambda x: x["category"] == "technical",
        technical_chain
    ),

    general_chain
)

result = branch_chain.invoke({
    "category": "technical",
    "question": "My application is not starting."
})

print(result)

branch_chain.get_graph().print_ascii()