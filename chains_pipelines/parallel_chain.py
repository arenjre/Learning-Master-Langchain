from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


from dotenv import load_dotenv
load_dotenv()

prompt1 = PromptTemplate(
    template="""
You are a technology expert.
Analyze the following product and explain its main advantages.

Product: {product}

Give 5 advantages with short explanations.
""",
    input_variables=["product"]
)

prompt2 = PromptTemplate(
    template="""
You are a technology expert.
Analyze the following product and explain its main disadvantages.

Product: {product}

Give 5 disadvantages with short explanations.
""",
    input_variables=["product"]
)

prompt3 = PromptTemplate(
    template="""
You are a technology advisor.

Using the advantages and disadvantages below, create a final product analysis.

Advantages:
{advantages}

Disadvantages:
{disadvantages}

Provide:
1. Short summary
2. Key advantages
3. Key disadvantages
4. Who should buy it
5. Final recommendation
""",
    input_variables=["advantages", "disadvantages"]
)

parser = StrOutputParser()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation",
    max_new_tokens=1024,
)

model = ChatHuggingFace(llm=llm) 

parallel_chain = RunnableParallel({
    "advantages": prompt1 | model | parser,
    "disadvantages": prompt2 | model | parser
})

merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain

result = chain.invoke({
    "product": "iPhone 17 Pro"
})

print(result)
chain.get_graph().print_ascii()