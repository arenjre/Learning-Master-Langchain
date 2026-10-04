from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from dotenv import load_dotenv
load_dotenv()

parser = StrOutputParser()

template1 = PromptTemplate(
    template="You are a professional teacher. Please review the following topic and provide details for the students: {topic}",
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template="You are a professional teacher, make a short multiple choice quiz for the following text and provide answers: {text}",
    input_variables=["text"],
)

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

chain = template1 | model | parser | template2 | model | parser

# print(chain.invoke({"topic": "Python programming language"}))
print(chain.get_graph().draw_ascii())