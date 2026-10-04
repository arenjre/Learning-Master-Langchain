# from langchain_community.tools import DuckDuckGoSearchRun

# tool = DuckDuckGoSearchRun()

# result = tool.invoke("Latest Generative AI trends")
# print(result)

# from langchain_community.tools import WikipediaQueryRun
# from langchain_community.utilities import WikipediaAPIWrapper

# tool = WikipediaQueryRun(
#     api_wrapper=WikipediaAPIWrapper()
# )

# result = tool.invoke("Artificial Intelligence")
# print(result)


# from langchain_community.tools import ShellTool

# tool = ShellTool()

# result = tool.invoke("echo Hello from LangChain")
# print(result)

from langchain_community.tools import ReadFileTool

tool = ReadFileTool()

result = tool.invoke("example_tools.txt")
print(result)

from langchain_community.tools import WriteFileTool

tool = WriteFileTool()

result = tool.invoke({
    "file_path": "example.txt",
    "text": "Hello Anand!\nI am learning LangChain."
})

print(result)