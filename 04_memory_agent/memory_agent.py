"""
STEP 4: Memory & Context — An Agent That Remembers
====================================================

WHAT YOU'LL LEARN:
- Short-term memory (conversation history)
- Long-term memory (saving/loading notes from a file)
- Context window limits and why they matter
- Conversation summarization (compressing old messages)

THE MEMORY PROBLEM:
--------------------
LLMs have NO built-in memory. Every API call starts fresh.
We fake "memory" by sending previous messages along with new ones.

But there's a limit — LLMs can only read a certain amount of text
(called the "context window"). For example:
- GPT-4o: ~128,000 tokens (~96,000 words)
- Claude: ~200,000 tokens (~150,000 words)

If your conversation gets too long, it won't fit. Solutions:
1. Truncate old messages (lose information)
2. Summarize old messages (compress information)  <-- we'll do this
3. Use a vector database to store and search (Step 5 territory)

TWO TYPES OF MEMORY WE'LL BUILD:
----------------------------------
1. SHORT-TERM: The conversation history list (same as before)
   - Lives in RAM, lost when you close the program

2. LONG-TERM: A JSON file on disk
   - The agent can save notes like "User's name is Sujit"
   - Next time you start the agent, it loads these notes
   - Persists across sessions
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

# Path to the long-term memory file (persists across sessions)
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "memory.json")


# ===========================================================================
# PART 1: LONG-TERM MEMORY — Save/Load from a file
# ===========================================================================

def load_long_term_memory() -> dict:
    """Load saved memories from disk. Returns empty dict if no file exists."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {"notes": [], "user_info": {}}


def save_long_term_memory(memory: dict):
    """Save memories to disk so they persist across sessions."""
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


def format_long_term_memory(memory: dict) -> str:
    """Format saved memories into text the LLM can read."""
    parts = []

    if memory["user_info"]:
        parts.append("Known facts about the user:")
        for key, value in memory["user_info"].items():
            parts.append(f"  - {key}: {value}")

    if memory["notes"]:
        parts.append("\nSaved notes:")
        for note in memory["notes"][-10:]:  # only last 10 notes
            parts.append(f"  - [{note['timestamp']}] {note['content']}")

    return "\n".join(parts) if parts else "No saved memories yet."


# ===========================================================================
# PART 2: TOOLS — Including memory tools
# ===========================================================================

# Load long-term memory at startup
long_term_memory = load_long_term_memory()


def save_note(content: str) -> str:
    """Save a note to long-term memory."""
    note = {
        "content": content,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    long_term_memory["notes"].append(note)
    save_long_term_memory(long_term_memory)
    return f"Saved note: '{content}'"


def save_user_info(key: str, value: str) -> str:
    """Save a fact about the user (name, preferences, etc.)."""
    long_term_memory["user_info"][key] = value
    save_long_term_memory(long_term_memory)
    return f"Remembered: {key} = {value}"


def recall_memories() -> str:
    """Recall all saved long-term memories."""
    return format_long_term_memory(long_term_memory)


def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")


def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters"
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


# Tool definitions for the API
tools = [
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a note to long-term memory. Use this when the user asks you to remember something, or when important information comes up that should be saved for later.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The note content to save"
                    }
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_user_info",
            "description": "Save a fact about the user like their name, preferences, or interests. Use this when the user tells you personal details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The category (e.g. 'name', 'favorite_language', 'skill_level')"
                    },
                    "value": {
                        "type": "string",
                        "description": "The value to remember"
                    }
                },
                "required": ["key", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memories",
            "description": "Recall all saved long-term memories (notes and user info). Use this when you need to check what you know about the user or what was previously saved.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

tool_functions = {
    "save_note": save_note,
    "save_user_info": save_user_info,
    "recall_memories": recall_memories,
    "get_current_time": get_current_time,
    "calculator": calculator,
}


# ===========================================================================
# PART 3: CONVERSATION SUMMARIZATION
# ===========================================================================
# When the conversation gets long, we ask the LLM to summarize older messages.
# This compresses the history so it fits in the context window.

MAX_MESSAGES = 20  # After this many messages, summarize older ones


def summarize_conversation(messages: list) -> list:
    """
    Compress older messages into a summary.

    Takes the first N messages (excluding system prompt) and asks the LLM
    to summarize them into a single message. This frees up space for new messages.
    """
    if len(messages) <= MAX_MESSAGES:
        return messages  # not too long yet

    print("\n  [System] Conversation getting long, summarizing older messages...")

    # Split: keep system prompt + summarize middle + keep recent
    system_msg = messages[0]
    old_messages = messages[1:-8]   # messages to summarize
    recent_messages = messages[-8:]  # keep last 8 messages as-is

    # Build summary text from old messages
    summary_text = ""
    for msg in old_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            speaker = "User" if role == "user" else "Assistant"
            summary_text += f"{speaker}: {content}\n"

    if not summary_text.strip():
        return [system_msg] + recent_messages

    # Ask the LLM to summarize
    summary_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Summarize this conversation in 2-3 sentences. Keep key facts and decisions."},
            {"role": "user", "content": summary_text},
        ],
        max_completion_tokens=200,
    )

    summary = summary_response.choices[0].message.content

    # Build new message list with summary replacing old messages
    summarized = [
        system_msg,
        {"role": "system", "content": f"[Summary of earlier conversation: {summary}]"},
    ] + recent_messages

    print(f"  [System] Compressed {len(old_messages)} messages into a summary.")
    return summarized


# ===========================================================================
# PART 4: THE AGENT LOOP (same as Step 2, but with memory)
# ===========================================================================

def build_system_prompt() -> str:
    """Build system prompt that includes long-term memories."""
    base = """You are a helpful assistant with memory capabilities.

You can remember things across conversations using your memory tools.
When the user tells you personal details (name, preferences, etc.), save them using save_user_info.
When the user asks you to remember something, use save_note.
When you need to recall what you know, use recall_memories.

Be conversational and friendly. If you have saved memories about the user, use them naturally.
"""
    # Inject long-term memories into the system prompt
    memories = format_long_term_memory(long_term_memory)
    if memories != "No saved memories yet.":
        base += f"\n\nYour saved memories:\n{memories}\n"

    return base


# Initialize conversation with system prompt that includes memories
conversation_history = [
    {"role": "system", "content": build_system_prompt()}
]


def run_agent(user_message: str) -> str:
    """Agent loop with tool calling (same pattern as Step 2)."""
    global conversation_history

    conversation_history.append({"role": "user", "content": user_message})

    # Summarize if conversation is getting too long
    conversation_history = summarize_conversation(conversation_history)

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=conversation_history,
            tools=tools,
            max_completion_tokens=1000,
        )

        message = response.choices[0].message

        if message.tool_calls:
            conversation_history.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"  [Memory] {tool_name}({tool_args})")

                func = tool_functions.get(tool_name)
                if func:
                    result = func(**tool_args) if tool_args else func()
                else:
                    result = f"Unknown tool: {tool_name}"

                print(f"  [Memory] Result: {result}")

                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
            continue

        conversation_history.append({"role": "assistant", "content": message.content})
        return message.content


# ===========================================================================
# PART 5: MAIN
# ===========================================================================

def main():
    print("=" * 55)
    print("  Memory Agent (Step 4)")
    print("  An agent that remembers across sessions!")
    print("  Type 'quit' to exit")
    print("  Type 'memory' to see saved memories")
    print("=" * 55)

    # Show existing memories on startup
    memories = format_long_term_memory(long_term_memory)
    if memories != "No saved memories yet.":
        print(f"\n  Loaded memories from previous session:")
        for line in memories.split("\n"):
            print(f"    {line}")
    else:
        print("\n  No previous memories found. Start chatting!")

    print()
    print("  Try these:")
    print("    'My name is Sujit and I am learning AI'")
    print("    'Remember that I prefer Python over JavaScript'")
    print("    'What do you know about me?'")
    print()

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye! I'll remember our conversation.")
            break
        if user_input.lower() == "memory":
            print(f"\n{format_long_term_memory(long_term_memory)}\n")
            continue

        try:
            response = run_agent(user_input)
            safe = response.encode("utf-8", errors="replace").decode("utf-8")
            print(f"\nAssistant: {safe}\n")
        except Exception as e:
            print(f"\nError: {e}\n")
            conversation_history.pop()


if __name__ == "__main__":
    main()
