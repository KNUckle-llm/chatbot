from langgraph.checkpoint.memory import InMemorySaver  # short-term memory
from langgraph.store.memory import InMemoryStore  # long-term memory
from langgraph.graph import StateGraph, START, END

from .state import CustomState
from .nodes import answer_node


def build_graph():
    checkpointer = InMemorySaver()  # short-term memory
    # store = InMemoryStore()  # long-term memory
    builder = StateGraph(CustomState)
    builder.add_node("answer", answer_node)

    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)

    graph = builder.compile(checkpointer=checkpointer,)
    return graph
