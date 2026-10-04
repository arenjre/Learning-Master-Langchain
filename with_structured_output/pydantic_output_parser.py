from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from pydantic_models import PersonInfo

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

load_dotenv()

parser = PydanticOutputParser(pydantic_object=PersonInfo)

template = PromptTemplate(
    template = "Extract the relevant information from the query.\n{format_instructions}\nQuestion: {question}",
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

chain = template | model | parser

user_input = input("Enter your question: ")

try:
    response = chain.invoke({"question": user_input})
    print("\n Successfully Parsed:")
    print(response)

except OutputParserException as exc:
    print("\n  [Validation Warning] Output schema validation(s) failed:")
    
    # Original Pydantic ValidationError extract karna
    if isinstance(exc.__cause__, ValidationError):
        for err in exc.__cause__.errors():
            field_name = " -> ".join(str(loc) for loc in err["loc"])
            reason = err["msg"]
            given_value = err.get("input", "N/A")
            
            print(f"  • Field: '{field_name}' | Value: '{given_value}' | Issue: {reason}")
    else:
        print(f"  • Parsing failed: {exc}")

except Exception as exc:
    print(f"\n  [Unexpected Error]: {exc}")


