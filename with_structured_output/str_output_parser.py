from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template="You are a helpful assistant. Answer the following question: {question}",
    input_variables=["question"]
)

parser = StrOutputParser()

# Chain order: Prompt -> Model -> Output Parser
input = input("Enter your question: ")
result = template | model | parser

print(result.invoke({"question": input}))
