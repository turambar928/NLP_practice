from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-mini", output_version="responses/v1")

tool = {"type": "image_generation", "quality": "low"}

llm_with_tools = llm.bind_tools([tool])

ai_message = llm_with_tools.invoke(
    "Draw a picture of a cute fuzzy cat with an umbrella"
)
