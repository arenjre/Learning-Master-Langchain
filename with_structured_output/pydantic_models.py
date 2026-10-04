from pydantic import BaseModel, Field

class PersonInfo(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person", gt=18, default=00)
    city: str = Field(description="Name of the city")
    profession: str = Field(description="Profession or occupation of the person")