# type dict
from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    email: str

# p1 = Person(name="Alice", age="30", email="demo")
# print(p1)
# it make structure but it does now have any data type validation, see in email we put "demo" and in age we put 30 in string it saved 


from typing import TypedDict, Optional, Annotated, Literal
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

_review = "Despite solid build quality and snappy app switching with 12GB RAM, the thermal throttling under high gaming loads and inconsistent fast-charging speeds hold this device back"

_review2 = '''
The phone delivers a remarkably polished user experience, headlined by its vibrant 120Hz AMOLED display that makes scrolling and media consumption exceptionally fluid. Powered by a efficient processor alongside 12GB of RAM, app switching feels instantaneous, and day-to-day multitasking happens without a single hitch. Build quality feels premium in the hand with its matte glass back, and the primary 50MP camera captures sharp, color-accurate photos in daylight conditions. Battery performance easily stretches past a full day of heavy usage, backed by fast wired charging that gets you back to full power quickly.

The device is not without its limitations, however. Under sustained heavy workloads like 3D gaming, noticeable thermal throttling kicks in, causing temporary drops in frame rates. Nighttime camera performance trails behind daytime results, often introducing visual noise in low-light environments. Additionally, the lack of an expandable microSD card slot limits local storage upgrades, and the pre-installed bloatware requires a few extra minutes of cleanup right out of the box
'''

# schema
class Review(TypedDict):
    pros: Annotated[list[str], "A list of positive aspects mentioned in the review"]
    cons: Annotated[list[str], "A list of negative aspects mentioned in the review"]
    summary: Annotated[str, "A brief summary of the review, in 1-2 sentences"]
    # sentiment: Annotated[Literal["positive", "negative", "neutral"], ..., "The overall sentiment of the review, either 'positive', 'negative', or 'neutral'"]
    sentiment: Annotated[Literal['Positive', 'Negative', 'Neutral'], "The overall sentiment of the review, either 'Positive', 'Negative', or 'Neutral'"]
    key_thems: Annotated[list[str], "Add all key themes which are discussed in the review"]

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

structured_output = model.with_structured_output(Review)

result = structured_output.invoke(_review2)
print(result)