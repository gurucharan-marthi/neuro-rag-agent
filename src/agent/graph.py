"""LangGraph agentic RAG: retrieve -> grade -> (PubMed tool if needed) -> generate."""
import json
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from src.agent.tools import pubmed_search, search_local

load_dotenv()
import os
llm = ChatGroq(model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"), temperature=0)


# ---------- State ----------
class State(TypedDict):
    question: str
    docs: list[dict]
    messages: Annotated[list, add_messages]   # needed for tool calling
    answer: str


# ---------- Nodes ----------
def retrieve(state: State):
    return {"docs": search_local(state["question"])}


class Grade(BaseModel):
    relevant: bool = Field(description="True if the paper helps answer the question")

grader = llm.with_structured_output(Grade)

def grade(state: State):
    kept = []
    for d in state["docs"]:
        g = grader.invoke(
            f"Question: {state['question']}\n\n"
            f"Paper: {d['title']}. {d['abstract'][:1500]}\n\n"
            "Does this paper help answer the question?"
        )
        if g.relevant:
            kept.append(d)
    print(f"[grade] {len(kept)}/{len(state['docs'])} local papers relevant")
    return {"docs": kept}


research_llm = llm.bind_tools([pubmed_search])

def research(state: State):
    msg = research_llm.invoke([
        SystemMessage("The local paper database lacked relevant papers. "
                      "Call pubmed_search with a concise keyword query "
                      "to find papers that answer the user's question."),
        HumanMessage(state["question"]),
    ])
    return {"messages": [msg]}


def collect(state: State):
    new = []
    for m in state["messages"]:
        if m.type == "tool":
            new.extend(json.loads(m.content))
    print(f"[collect] {len(new)} papers from PubMed")
    return {"docs": state["docs"] + new}


def generate(state: State):
    if not state["docs"]:
        return {"answer": "I couldn't find relevant papers to answer this."}
    context = "\n\n".join(
        f"[{i+1}] {d['title']} ({d['url']})\n{d['abstract'][:1500]}"
        for i, d in enumerate(state["docs"])
    )
    prompt = (
        "Answer the question using ONLY the papers below. "
        "Cite papers as [1], [2]. If the papers don't contain the answer, say so.\n\n"
        f"Papers:\n{context}\n\nQuestion: {state['question']}"
    )
    return {"answer": llm.invoke(prompt).content}


# ---------- Routing ----------
def route_after_grade(state: State):
    return "generate" if len(state["docs"]) >= 2 else "research"


# ---------- Build graph ----------
def build_graph():
    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("research", research)
    g.add_node("tools", ToolNode([pubmed_search]))
    g.add_node("collect", collect)
    g.add_node("generate", generate)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade,
                            {"generate": "generate", "research": "research"})
    g.add_conditional_edges("research", tools_condition,
                            {"tools": "tools", END: "generate"})
    g.add_edge("tools", "collect")
    g.add_edge("collect", "generate")
    g.add_edge("generate", END)
    return g.compile()


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "How is the optic chiasm segmented in MRI?"
    out = build_graph().invoke({"question": q, "docs": [], "messages": []})
    print("\n=== ANSWER ===\n" + out["answer"])
    print("\n=== SOURCES ===")
    for i, d in enumerate(out["docs"]):
        print(f"[{i+1}] ({d['source']}) {d['title']} - {d['url']}")
