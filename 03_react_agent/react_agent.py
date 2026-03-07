"""
STEP 3: ReAct Agent — Reasoning + Acting
==========================================

WHAT YOU'LL LEARN:
- The ReAct pattern (Thought → Action → Observation → repeat)
- How an agent reasons through MULTI-STEP problems
- How to build an agent loop using plain text (no special API features)
- Why showing reasoning makes agents more reliable

WHAT'S DIFFERENT FROM STEP 2?
-------------------------------
Step 2 used the API's built-in tool_calls feature.
Step 3 builds the agent loop using ONLY text prompts.

Why? Because:
1. It shows you how agents REALLY work under the hood
2. The original ReAct paper used this text-based approach
3. Not all LLMs support tool_calls — but ALL LLMs can output text
4. You can see the AI's reasoning, which helps with debugging

THE REACT LOOP:
    Thought  → The AI explains what it's thinking
    Action   → The AI picks a tool and inputs
    Observation → Your code runs the tool, gives the result
    ... repeat until ...
    Final Answer → The AI gives the final response
"""

import os
import re
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

client = OpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT") + "/openai/v1",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)
MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-5.2-chat")


# ===========================================================================
# PART 1: TOOLS — more interesting ones for multi-step reasoning
# ===========================================================================

# A simple knowledge base (simulates searching the internet)
KNOWLEDGE_BASE = {
    "india population": "India has a population of approximately 1.44 billion (2024).",
    "japan population": "Japan has a population of approximately 123 million (2024).",
    "usa population": "The United States has a population of approximately 335 million (2024).",
    "china population": "China has a population of approximately 1.41 billion (2024).",
    "python language": "Python is a high-level programming language created by Guido van Rossum in 1991. It is known for its simple syntax and is widely used in AI/ML, web development, and data science.",
    "javascript language": "JavaScript is a programming language created by Brendan Eich in 1995. It is the main language of the web and runs in browsers.",
    "eiffel tower height": "The Eiffel Tower is 330 meters (1,083 feet) tall.",
    "mount everest height": "Mount Everest is 8,849 meters (29,032 feet) tall.",
    "speed of light": "The speed of light is approximately 299,792,458 meters per second.",
    "earth distance sun": "The average distance from Earth to the Sun is about 149.6 million kilometers.",
}


def search(query: str) -> str:
    """Search the knowledge base for information."""
    query_lower = query.lower().strip()
    # Try exact match first
    if query_lower in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[query_lower]
    # Try partial match
    for key, value in KNOWLEDGE_BASE.items():
        if query_lower in key or key in query_lower:
            return value
    return f"No results found for '{query}'. Available topics: {', '.join(KNOWLEDGE_BASE.keys())}"


def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return f"Error: Invalid characters in expression"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")


# Map of tool names to functions
TOOLS = {
    "search": search,
    "calculator": calculator,
    "get_current_time": get_current_time,
}


# ===========================================================================
# PART 2: THE REACT PROMPT — This is where the magic happens
# ===========================================================================
# We tell the LLM exactly how to format its thinking.
# The LLM will output text like:
#   Thought: I need to find India's population
#   Action: search
#   Action Input: india population
#
# Our code will PARSE this text and run the tool.

REACT_SYSTEM_PROMPT = """You are a helpful assistant that solves problems step by step.

You have access to the following tools:

1. search(query) - Search a knowledge base for facts. Use this to look up information.
   Available topics: populations, programming languages, landmarks, science facts.

2. calculator(expression) - Calculate math expressions like "1.44 / 0.123" or "(330 * 2) + 100"

3. get_current_time() - Get the current date and time

To use a tool, you MUST use this EXACT format:

Thought: [your reasoning about what to do next]
Action: [tool name - one of: search, calculator, get_current_time]
Action Input: [the input to the tool]

After you see the Observation (tool result), continue with another Thought.

When you have enough information to answer, use this EXACT format:

Thought: [your final reasoning]
Final Answer: [your complete answer to the user]

IMPORTANT RULES:
- Always start with a Thought
- Use ONE tool at a time
- After each Observation, write a new Thought
- When done, use "Final Answer:" (not another Action)
- For math, ALWAYS use the calculator tool, never calculate in your head
"""


# ===========================================================================
# PART 3: PARSE THE LLM'S OUTPUT
# ===========================================================================
# The LLM outputs plain text. We need to extract:
#   - Is it an Action? → which tool, what input
#   - Is it a Final Answer? → we're done

def parse_action(text: str):
    """
    Parse the LLM's output to find either an Action or a Final Answer.

    Returns:
        ("action", tool_name, tool_input) — if the LLM wants to use a tool
        ("answer", final_answer, None)    — if the LLM is giving the final answer
        ("error", message, None)          — if we can't parse the output
    """
    # Check for Final Answer
    final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if final_match:
        return ("answer", final_match.group(1).strip(), None)

    # Check for Action + Action Input
    action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text)
    input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text)

    if action_match:
        tool_name = action_match.group(1).strip().lower()
        tool_input = input_match.group(1).strip() if input_match else ""
        return ("action", tool_name, tool_input)

    return ("error", "Could not parse response. Raw output:\n" + text, None)


# ===========================================================================
# PART 4: THE REACT LOOP
# ===========================================================================

def run_react_agent(user_question: str, max_steps: int = 8) -> str:
    """
    Run the ReAct loop:
        Thought → Action → Observation → Thought → ... → Final Answer

    We build up a single text "scratchpad" that contains the full
    reasoning trace. This gets sent to the LLM each time so it can
    see its own previous thoughts and observations.
    """

    print(f"\n{'='*55}")
    print(f"  Question: {user_question}")
    print(f"{'='*55}")

    # The scratchpad holds the full Thought/Action/Observation trace
    # Think of it as the agent's "working notes"
    scratchpad = ""

    for step in range(1, max_steps + 1):
        print(f"\n--- Step {step} ---")

        # Build the prompt: system instructions + user question + scratchpad so far
        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {user_question}\n\n{scratchpad}"},
        ]

        # Ask the LLM
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_completion_tokens=500,
        )

        llm_output = response.choices[0].message.content
        print(f"{llm_output}")

        # Parse: is it an action or a final answer?
        result_type, value, tool_input = parse_action(llm_output)

        if result_type == "answer":
            print(f"\n{'='*55}")
            return value

        elif result_type == "action":
            # Run the tool
            tool_name = value
            if tool_name in TOOLS:
                if tool_input:
                    observation = TOOLS[tool_name](tool_input)
                else:
                    observation = TOOLS[tool_name]()
            else:
                observation = f"Error: Unknown tool '{tool_name}'. Available: {', '.join(TOOLS.keys())}"

            print(f"Observation: {observation}")

            # Add this step to the scratchpad
            scratchpad += llm_output + f"\nObservation: {observation}\n\n"

        else:
            # Parse error — add raw output and let LLM try again
            scratchpad += llm_output + "\n"

    return "Agent stopped: too many steps without reaching an answer."


# ===========================================================================
# PART 5: MAIN
# ===========================================================================

def main():
    print("=" * 55)
    print("  ReAct Agent (Step 3)")
    print("  Think → Act → Observe → Repeat")
    print("  Type 'quit' to exit")
    print("=" * 55)
    print()
    print("  Try multi-step questions like:")
    print("    'How many times taller is Everest than the Eiffel Tower?'")
    print("    'Which has more people, India or China? By how much?'")
    print("    'What is the population of USA plus Japan?'")
    print()

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        try:
            answer = run_react_agent(user_input)
            safe_answer = answer.encode("utf-8", errors="replace").decode("utf-8")
            print(f"\nFinal Answer: {safe_answer}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
