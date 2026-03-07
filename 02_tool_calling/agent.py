"""
STEP 2: Tool Calling — Your First Agent
========================================

WHAT YOU'LL LEARN:
- How to define "tools" that an LLM can use
- How the LLM decides WHICH tool to call (and with what inputs)
- The tool calling loop: Ask → LLM picks tool → Run tool → Give result back → Repeat

HOW TOOL CALLING WORKS (the key insight):
------------------------------------------

1. You describe your tools to the LLM in a specific JSON format:
   "There's a tool called 'calculator' that takes two numbers and an operation"

2. When the user asks something, the LLM can either:
   a) Reply normally (if no tool is needed)
   b) Say "I want to call the 'calculator' tool with these arguments"

3. The LLM does NOT run the tool. It just outputs a structured request.
   YOUR CODE runs the tool and sends the result back.

4. The LLM then reads the result and decides:
   a) Reply to the user (if it has enough info)
   b) Call ANOTHER tool (if it needs more info)

This is the foundation of ALL AI agents.
"""

import os
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
# PART 1: DEFINE THE TOOLS (Python functions)
# ===========================================================================
# These are regular Python functions. The AI doesn't see this code.
# The AI only sees the DESCRIPTION we give it (in the tools list below).

def calculator(expression: str) -> str:
    """Evaluate a math expression safely."""
    try:
        # Only allow safe math characters
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return f"Error: Invalid characters in expression"
        result = eval(expression)  # safe because we filtered characters
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_current_time() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S (%A)")


def word_count(text: str) -> str:
    """Count words in the given text."""
    words = text.split()
    return f"{len(words)} words"


# ===========================================================================
# PART 2: DESCRIBE THE TOOLS TO THE LLM
# ===========================================================================
# This is the JSON "menu" we show the LLM. It tells the LLM:
#   - What tools exist
#   - What each tool does (description)
#   - What inputs each tool needs (parameters)
#
# The LLM reads these descriptions and decides which tool fits the user's request.
# Think of it as a restaurant menu — the LLM reads it and places an order.

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression. Use this for any calculations, arithmetic, or math problems.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '2 + 3 * 4' or '(10 + 5) / 3'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time. Use this when the user asks about today's date or current time.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "word_count",
            "description": "Count the number of words in a given text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to count words in"
                    }
                },
                "required": ["text"]
            }
        }
    }
]

# ===========================================================================
# PART 3: MAP TOOL NAMES TO ACTUAL FUNCTIONS
# ===========================================================================
# When the LLM says "call calculator", we need to know which Python function to run.
# This dictionary connects the name (string) to the actual function.

tool_functions = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "word_count": word_count,
}


# ===========================================================================
# PART 4: THE AGENT LOOP (this is the core of every agent)
# ===========================================================================

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
Use tools when they would help answer the user's question.
For math, ALWAYS use the calculator tool instead of calculating yourself.
"""

conversation_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]


def run_agent(user_message: str) -> str:
    """
    The agent loop. This is what makes it an AGENT instead of a chatbot.

    CHATBOT: User → LLM → Reply (one step)
    AGENT:   User → LLM → Tool Call → Result → LLM → (maybe more tools) → Reply

    The loop keeps going until the LLM decides to reply normally
    (instead of calling another tool).
    """

    conversation_history.append({"role": "user", "content": user_message})

    while True:
        # ------------------------------------------------------------------
        # Step A: Ask the LLM what to do
        # ------------------------------------------------------------------
        # We send the conversation + tool descriptions.
        # The LLM will either:
        #   - Reply with text (task is done)
        #   - Request a tool call (needs more info)
        response = client.chat.completions.create(
            model=MODEL,
            messages=conversation_history,
            tools=tools,             # <-- THIS IS NEW! We pass the tool menu
            max_completion_tokens=1000,
        )

        message = response.choices[0].message

        # ------------------------------------------------------------------
        # Step B: Check if the LLM wants to call tools
        # ------------------------------------------------------------------
        if message.tool_calls:
            # The LLM wants to use one or more tools!
            # Add the LLM's response (with tool call info) to history
            conversation_history.append(message)

            # Process each tool call
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"  [Agent] Calling tool: {tool_name}({tool_args})")

                # Actually run the tool
                if tool_name in tool_functions:
                    if tool_args:
                        result = tool_functions[tool_name](**tool_args)
                    else:
                        result = tool_functions[tool_name]()
                else:
                    result = f"Error: Unknown tool '{tool_name}'"

                print(f"  [Agent] Tool result: {result}")

                # Send the tool result back to the LLM
                # The role "tool" tells the LLM "this is the output of the tool you called"
                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            # Loop back to Step A — the LLM will read the tool results
            # and decide whether to call more tools or give a final reply
            continue

        # ------------------------------------------------------------------
        # Step C: No tool calls — the LLM is giving a final text reply
        # ------------------------------------------------------------------
        conversation_history.append({"role": "assistant", "content": message.content})
        return message.content


# ===========================================================================
# PART 5: MAIN LOOP
# ===========================================================================

def main():
    print("=" * 55)
    print("  AI Agent with Tools (Step 2)")
    print("  Tools: calculator, get_current_time, word_count")
    print("  Type 'quit' to exit")
    print("=" * 55)
    print()
    print("  Try these:")
    print("    'What is 1547 * 382?'")
    print("    'What day is it today?'")
    print("    'Count words in: the quick brown fox jumps'")
    print()

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        try:
            response = run_agent(user_input)
            # encode/decode to handle emojis on Windows console
            safe_response = response.encode("utf-8", errors="replace").decode("utf-8")
            print(f"\nAssistant: {safe_response}\n")
        except Exception as e:
            print(f"\nError: {e}\n")
            conversation_history.pop()


if __name__ == "__main__":
    main()
