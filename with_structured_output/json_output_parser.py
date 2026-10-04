from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="You are a helpful assistant. Provide details in JSON format for the question.\n{format_instructions}\nQuestion: {question}",
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser

# 5. Execute
user_input = input("Enter your question: ")
response = chain.invoke({"question": user_input})

print(response)
print(type(response))


# question = "What are the key features of the latest iPhone model?"

# output 
'''
{
  question: "What are the key features of the latest iPhone model?",
  product: "iPhone 15 Series",
  key_features: {
    design_and_display: [
      "Dynamic Island across all models",
      "Super Retina XDR display",
      "Titanium frame (Pro models) for reduced weight and increased durability",
      "Ceramic Shield front cover",
    ],
    camera_system: [
      "48MP Main camera (all models)",
      "5x Optical Zoom (Pro Max model)",
      "Next-generation portraits with focus and depth control",
      "Photonic Engine for improved mid-to-low light performance",
    ],
    performance: [
      "A17 Pro chip (Pro models) with hardware-accelerated ray tracing",
      "A16 Bionic chip (Standard models)",
      "Enhanced GPU for high-end gaming",
    ],
    connectivity_and_charging: [
      "USB-C connector replacing Lightning",
      "USB 3 speeds for faster data transfer (Pro models)",
      "Second-generation Ultra Wideband chip",
    ],
    safety_and_emergency: [
      "Emergency SOS via satellite",
      "Crash Detection",
      "Roadside Assistance via satellite",
    ],
  },
}
'''