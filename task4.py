from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_agent
from dotenv import load_dotenv
import os

# Load API keys
load_dotenv()

# Initialize LLM (Ollama must be running)
llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)


# -------------------- TOOLS --------------------

@tool
def web_search(query: str) -> str:
    """Search the web for real time information on any topic"""
    search = TavilySearchResults(max_results=4)
    results = search.invoke({"query": query})

    output = ""
    for r in results:
        output += r['content']
    return output.strip()


@tool
def summarise(text: str) -> str:
    """Summarise long text into 2-3 sentences"""
    response = llm.invoke(
        f"Summarize the following in 2-3 sentences:\n\n{text}"
    )
    return response.content


@tool
def notes_taker(content: str) -> str:
    """Convert text into structured notes with title and bullet points"""

    prompt = f"""
    Convert the following into structured notes.

    Rules:
    - Create a short title
    - Use bullet points
    - Keep it concise

    Format:
    Title: ...
    Content:
    - point 1
    - point 2

    Content:
    {content}
    """

    result = llm.invoke(prompt)
    # Generate filename using timestamp
    filename = f"notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # Save to file
    with open(filename, "w", encoding="utf-8") as file:
        file.write(notes)

    return f"Notes saved successfully as {filename}\n\n{notes}"


# -------------------- AGENT --------------------

tools = [web_search, summarise, notes_taker]

agent = create_agent(
    llm,
    tools,
    system_prompt="""
    You are a helpful AI assistant.

    When answering:
    - Use web_search for real-time or latest information
    - Use summarise for long text
    - Use notes_taker to format final answers into notes

    Try to use tools whenever helpful.
    """
)


# -------------------- RUNNER --------------------

def run_react_agent(query):
    """Run a query through the agent and show step-by-step output"""

    print(f"\nQuery: {query}")
    print("-" * 50)

    result = agent.invoke({
        "messages": [("human", query)]
    })

    for msg in result["messages"]:

        # Tool Calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"[Tool Call] {tc['name']}({tc['args']})")

        # Tool Results
        elif msg.type == "tool":
            print(f"[Tool Result from {msg.name}]")
            print(msg.content[:300])
            print("-" * 30)

        # Final Answer
        elif msg.type == "ai" and msg.content:
            print(f"[AI]\n{msg.content}")


# -------------------- RUN --------------------

if __name__ == "__main__":
    run_react_agent("latest news about AI and convert into notes")
