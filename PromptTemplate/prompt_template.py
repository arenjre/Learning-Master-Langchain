from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    imput_veriables=["topic"],
    template="Make a short note on {topic}"
)

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",  # or "Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"
    task="text-generation",
    # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)
model = ChatHuggingFace(llm=llm)
chain = prompt | model
result = chain.invoke({"topic": "Python programming"})
print(result.content)