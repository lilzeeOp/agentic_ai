"""
STEP 6: LangGraph — Building Agents with a Framework
======================================================

WHAT YOU'LL LEARN:
- What LangGraph is and why people use it
- How to define a graph (nodes + edges)
- Conditional routing (different paths based on decisions)
- How LangGraph replaces the manual loops we wrote in Steps 2-5

WHAT IS LANGGRAPH?
-------------------
LangGraph lets you build agents as GRAPHS (flowcharts).

Instead of writing while loops and if/else chains manually,
you define:
  - NODES: functions that do work (research, write, etc.)
  - EDGES: arrows connecting nodes (what runs after what)
  - STATE: a shared dictionary that flows through the graph

Think of it as drawing a flowchart, then LangGraph runs it for you.

OUR GRAPH:
-----------

    [Start]
       |
    [Router]  ← decides: simple question or complex task?
      / \
     /   \
 [Simple]  [Research]
    |         |
    |      [Write]
    |         |
    |      [Add Code] ← only if code is needed
    |         |
      \\       /
       \\     /
     [Final Output]
         |
       [End]

COMPARED TO WHAT WE BUILT MANUALLY:
-------------------------------------
Step 5 (manual):  while loop + if/else + tool_calls
Step 6 (LangGraph): define nodes + edges, framework runs the loop
"""

import os
import operator
from typing import Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# We use TypedDict from typing_extensions for the state schema
from typing_extensions import TypedDict

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


# ===========================================================================
# PART 1: SETUP THE LLM
# ===========================================================================
# LangChain wraps the OpenAI client in a "ChatOpenAI" object.
# This is the same LLM we used before, just wrapped differently.

llm = ChatOpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT") + "/openai/v1",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model=os.getenv("AZURE_OPENAI_MODEL", "gpt-5.2-chat"),
    max_tokens=1500,
)


# ===========================================================================
# PART 2: DEFINE THE STATE
# ===========================================================================
# The "state" is a dictionary that flows through every node in the graph.
# Each node can READ from the state and WRITE to it.
# Think of it as a shared notebook that every team member can use.

class AgentState(TypedDict):
    question: str               # The user's original question
    question_type: str          # "simple" or "complex" (decided by router)
    needs_code: bool            # Does the answer need code examples?
    research: str               # Output from the research node
    written_content: str        # Output from the writer node
    code_content: str           # Output from the coder node
    final_answer: str           # The combined final output


# ===========================================================================
# PART 3: DEFINE THE NODES (each node is a function)
# ===========================================================================
# Each node takes the state, does some work, and returns updates to the state.
# LangGraph automatically merges the updates into the state.

def router_node(state: AgentState) -> dict:
    """
    NODE 1: Router — Decides if the question is simple or complex.

    Simple = can be answered directly (e.g., "What is a variable?")
    Complex = needs research + writing (e.g., "Write a guide about decorators")
    """
    print("\n  [Router] Analyzing question...")

    response = llm.invoke([
        SystemMessage(content="""Classify this question into exactly one category.

Reply with ONLY a JSON object, nothing else:
{"type": "simple" or "complex", "needs_code": true or false}

Rules:
- "simple": factual questions, definitions, short explanations
- "complex": requests for articles, guides, comparisons, tutorials, blog posts
- "needs_code": true if the answer should include code examples"""),
        HumanMessage(content=state["question"]),
    ])

    # Parse the LLM's classification
    import json
    try:
        text = response.content.strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        classification = json.loads(text)
        q_type = classification.get("type", "simple")
        needs_code = classification.get("needs_code", False)
    except (json.JSONDecodeError, AttributeError):
        q_type = "simple"
        needs_code = False

    print(f"  [Router] Type: {q_type}, Needs code: {needs_code}")
    return {"question_type": q_type, "needs_code": needs_code}


def simple_answer_node(state: AgentState) -> dict:
    """
    NODE 2a: Simple Answer — For straightforward questions.
    Just answers directly without research/writing pipeline.
    """
    print("  [Simple] Answering directly...")

    response = llm.invoke([
        SystemMessage(content="""Give a clear, concise answer.
Keep it beginner-friendly. 2-3 paragraphs max.
If a code example would help, include one."""),
        HumanMessage(content=state["question"]),
    ])

    print("  [Simple] Done.")
    return {"final_answer": response.content}


def research_node(state: AgentState) -> dict:
    """
    NODE 2b: Research — Gathers facts about the topic.
    Same as our research_agent from Step 5, but now it's a graph node.
    """
    print("  [Researcher] Gathering facts...")

    response = llm.invoke([
        SystemMessage(content="""You are a research specialist.
Provide 5-7 key facts about the topic as bullet points.
Include specifics: numbers, dates, comparisons.
Keep it factual and organized."""),
        HumanMessage(content=f"Research this: {state['question']}"),
    ])

    print("  [Researcher] Done.")
    return {"research": response.content}


def writer_node(state: AgentState) -> dict:
    """
    NODE 3: Writer — Turns research into polished content.
    Receives the research results from the state.
    """
    print("  [Writer] Crafting content...")

    response = llm.invoke([
        SystemMessage(content="""You are a content writer for beginners.
Write clear, engaging content using the research provided.
Use short paragraphs, section headers, and a friendly tone.
Keep it under 300 words."""),
        HumanMessage(content=f"""User asked: {state['question']}

Research findings:
{state['research']}

Write a well-structured response using these findings."""),
    ])

    print("  [Writer] Done.")
    return {"written_content": response.content}


def coder_node(state: AgentState) -> dict:
    """
    NODE 4: Coder — Adds code examples (only runs if needs_code is True).
    """
    print("  [Coder] Writing code examples...")

    response = llm.invoke([
        SystemMessage(content="""You are a Python coding expert for beginners.
Write 1-2 short, well-commented code examples.
Use only Python standard library.
Keep each example under 20 lines.
Start with a brief explanation of what the code does."""),
        HumanMessage(content=f"""The user asked: {state['question']}

Here's the written content so far:
{state['written_content']}

Add relevant Python code examples that complement this content."""),
    ])

    print("  [Coder] Done.")
    return {"code_content": response.content}


def combine_node(state: AgentState) -> dict:
    """
    NODE 5: Combine — Merges writer + coder output into final answer.
    """
    print("  [Combiner] Assembling final output...")

    parts = [state.get("written_content", "")]
    if state.get("code_content"):
        parts.append("\n---\n## Code Examples\n")
        parts.append(state["code_content"])

    return {"final_answer": "\n".join(parts)}


# ===========================================================================
# PART 4: ROUTING FUNCTIONS (decide which path to take)
# ===========================================================================

def route_by_type(state: AgentState) -> str:
    """
    CONDITIONAL EDGE: After the router, go to either:
    - "simple_answer" if it's a simple question
    - "research" if it's a complex task
    """
    if state["question_type"] == "simple":
        return "simple_answer"
    return "research"


def route_after_writer(state: AgentState) -> str:
    """
    CONDITIONAL EDGE: After the writer, either:
    - Go to "coder" if code examples are needed
    - Go to "combine" if no code needed
    """
    if state.get("needs_code"):
        return "coder"
    return "combine"


# ===========================================================================
# PART 5: BUILD THE GRAPH
# ===========================================================================
# This is where we connect everything together.
# Think of it as drawing a flowchart.

def build_graph():
    """
    Build the agent graph:

        START → router → (simple?) → simple_answer → END
                       → (complex?) → research → writer → (code?) → coder → combine → END
                                                        → (no code) → combine → END
    """

    # Create a new graph with our state schema
    graph = StateGraph(AgentState)

    # --- Add nodes (the boxes in the flowchart) ---
    graph.add_node("router", router_node)
    graph.add_node("simple_answer", simple_answer_node)
    graph.add_node("research", research_node)
    graph.add_node("writer", writer_node)
    graph.add_node("coder", coder_node)
    graph.add_node("combine", combine_node)

    # --- Add edges (the arrows in the flowchart) ---

    # START → router (always)
    graph.add_edge(START, "router")

    # router → simple_answer OR research (conditional)
    graph.add_conditional_edges("router", route_by_type)

    # simple_answer → END
    graph.add_edge("simple_answer", END)

    # research → writer (always)
    graph.add_edge("research", "writer")

    # writer → coder OR combine (conditional)
    graph.add_conditional_edges("writer", route_after_writer)

    # coder → combine (always)
    graph.add_edge("coder", "combine")

    # combine → END
    graph.add_edge("combine", END)

    # Compile the graph (makes it runnable)
    return graph.compile()


# ===========================================================================
# PART 6: MAIN
# ===========================================================================

def main():
    print("=" * 55)
    print("  LangGraph Agent (Step 6)")
    print("  A graph-based agent with conditional routing")
    print("  Type 'quit' to exit")
    print("=" * 55)
    print()
    print("  Try these:")
    print("    Simple: 'What is a variable?'")
    print("    Complex: 'Write a guide about Python decorators'")
    print("    With code: 'Explain list comprehensions with examples'")
    print()

    # Build the graph once
    agent = build_graph()

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        try:
            # Run the graph with initial state
            print(f"\n  Running graph...")
            result = agent.invoke({
                "question": user_input,
                "question_type": "",
                "needs_code": False,
                "research": "",
                "written_content": "",
                "code_content": "",
                "final_answer": "",
            })

            safe = result["final_answer"].encode("utf-8", errors="replace").decode("utf-8")
            print(f"\n{'='*55}")
            print("  FINAL OUTPUT")
            print(f"{'='*55}")
            print(f"\n{safe}\n")

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
