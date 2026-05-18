import operator
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict


# Match the model family already used in the Groq-based Streamlit learning app.
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
SYSTEM_PROMPT = (
    "You are a helpful assistant tasked with performing arithmetic on a set of inputs. "
    "Use the provided tools for arithmetic whenever needed."
)


def build_model() -> ChatGroq:
    # Load local environment variables so the script can read GROQ_API_KEY.
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Set GROQ_API_KEY in .env or environment variables.")

    # Create the chat model once and let LangGraph drive when it is called.
    return ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        api_key=api_key,
    )


@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


# Register tools both as a list for binding and as a lookup map for execution.
tools = [add, multiply, divide]
tools_by_name = {registered_tool.name: registered_tool for registered_tool in tools}
model_with_tools = build_model().bind_tools(tools)


class MessagesState(TypedDict):
    # LangGraph will append new messages to the running conversation state.
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def llm_call(state: MessagesState) -> MessagesState:
    """LLM decides whether to call a tool or answer directly."""

    # The model sees the full running message history plus one fixed system instruction.
    response = model_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def tool_node(state: MessagesState) -> dict:
    """Execute tool calls requested by the model."""

    results = []
    # A single assistant message can request one or more tool calls.
    for tool_call in state["messages"][-1].tool_calls:
        selected_tool = tools_by_name[tool_call["name"]]
        observation = selected_tool.invoke(tool_call["args"])
        results.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        )
    return {"messages": results}


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Route to tools when the model requested a tool call; otherwise stop."""

    last_message = state["messages"][-1]
    # If tool_calls is present, the graph loops through the tool node first.
    if getattr(last_message, "tool_calls", None):
        return "tool_node"
    return END


def build_agent():
    # Build the minimal ReAct-style loop: model -> tools -> model.
    agent_builder = StateGraph(MessagesState)
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    agent_builder.add_edge("tool_node", "llm_call")
    return agent_builder.compile()


def run_agent(question: str) -> dict:
    # Start each run with one human message and an llm_calls counter at zero.
    agent = build_agent()
    return agent.invoke({"messages": [HumanMessage(content=question)], "llm_calls": 0})


def print_run_summary(result: dict) -> None:
    # ToolMessage entries prove that the arithmetic tools were actually executed.
    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    final_message = result["messages"][-1]

    print(f"LLM calls: {result.get('llm_calls', 0)}")
    print(f"Tool called: {'yes' if tool_messages else 'no'}")
    print("Final answer:")
    print(final_message.content)


if __name__ == "__main__":
    # Simple learning-lab example input.
    sample_question = "Add 3 and 4."
    print(f"Question: {sample_question}")
    result = run_agent(sample_question)
    print_run_summary(result)