# LangChain: PromptTemplate Overview

`PromptTemplate` ek simple text/string prompt banata hai.

## Code Example

```python
from langchain_core.prompts import PromptTemplate

# Prompt define karna
prompt = PromptTemplate(
    template="Explain {topic} in simple words.",
    input_variables=["topic"]
)

# Variable ke sath prompt ko invoke karna
result = prompt.invoke({
    "topic": "LangChain"
})

print(result)
```

## Generated Output

Isse roughly ye text banega:

```text
Explain LangChain in simple words.
```

---

## Role-based Context Representation

```yaml
System: You are a helpful assistant.
Human: Explain LangChain in simple words.
```