from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


from dotenv import load_dotenv
load_dotenv()


prompt = PromptTemplate(
    template="You are a helpful assistant. Please review the following text and provide feedback: {review}",
    input_variables=["review"]
)

parser = StrOutputParser()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

chain = prompt | model | parser

print(chain.invoke({"review": "This is a great product! I really enjoyed using it."}))
