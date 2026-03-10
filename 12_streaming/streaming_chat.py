"""
STEP 12: Streaming Agent — Real-Time Token-by-Token Output
============================================================

WHAT YOU'LL LEARN:
- How streaming works (get words as they're generated, not all at once)
- How to use stream=True with the OpenAI API
- How to process stream chunks (deltas)
- How to build a streaming chatbot
- How to add streaming to a tool-calling agent

HOW IT WORKS:
--------------
Without streaming:
  You ask → LLM thinks for 5 sec → BOOM full answer appears

With streaming:
  You ask → words appear one by one like typing → feels instant

The API sends tiny pieces called "chunks". Each chunk has a "delta"
which is just the next few characters the LLM generated.

You loop through chunks and print each delta immediately.

ONLY ONE CHANGE NEEDED:
  stream=True in the API call. That's it.
"""

import os
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

client = OpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT") + "/openai/v1",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)
MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-5.2-chat")


# ===========================================================================
# PART 1: BASIC STREAMING — Just a chatbot with typing effect
# ===========================================================================

def stream_chat(messages: list[dict]) -> str:
    """
    Send messages to LLM and STREAM the response.

    Instead of waiting for the full response, we get it piece by piece.

    Returns the full response text (so we can add it to conversation history).
    """

    # The ONLY difference: stream=True
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_completion_tokens=1000,
        stream=True,  # ← THIS IS THE MAGIC LINE
    )

    # response is now an ITERATOR of chunks, not a single response
    # We loop through each chunk as it arrives
    full_response = ""

    for chunk in response:
        # Some chunks (like the final usage stats) have no choices — skip them
        if not chunk.choices:
            continue

        # Each chunk has choices[0].delta (not .message)
        # delta contains just the NEW piece of text
        delta = chunk.choices[0].delta

        # delta.content is the new text piece (could be None at start/end)
        if delta.content:
            piece = delta.content
            # Print WITHOUT newline, flush immediately so it shows up
            sys.stdout.write(piece)
            sys.stdout.flush()
            full_response += piece

    # Print a newline at the end
    print()

    return full_response


# ===========================================================================
# PART 2: STREAMING WITH TOOLS — Agent that streams AND uses tools
# ===========================================================================

# Simple tools for demo
def get_weather(city: str) -> str:
    """Fake weather tool for demo."""
    weather_data = {
        "london": "15°C, Cloudy",
        "tokyo": "22°C, Sunny",
        "new york": "18°C, Partly Cloudy",
        "delhi": "35°C, Hot and Humid",
        "sydney": "20°C, Clear",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


def calculate(expression: str) -> str:
    """Simple calculator."""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters"
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate a math expression. Example: '(10 + 5) * 3'",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            }
        }
    }
]

tool_functions = {
    "get_weather": get_weather,
    "calculate": calculate,
}


def stream_agent(messages: list[dict]) -> str:
    """
    Agent loop with streaming.

    THE TRICKY PART:
    When streaming with tools, the tool call info arrives in chunks too.
    We need to collect all the chunks to build the complete tool call,
    then run the tool, then stream the final answer.

    Flow:
    1. Stream the response
    2. If it's a tool call → collect chunks → run tool → stream again
    3. If it's text → print it in real-time → done
    """

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            max_completion_tokens=1000,
            stream=True,
        )

        # We need to collect the full response from chunks
        # Because tool calls arrive in pieces too
        full_content = ""
        tool_calls_data = {}  # Collect tool call chunks here

        for chunk in response:
            # Skip chunks with no choices (usage stats, etc.)
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Case 1: Regular text content — print it immediately
            if delta.content:
                sys.stdout.write(delta.content)
                sys.stdout.flush()
                full_content += delta.content

            # Case 2: Tool call chunks — collect them
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_data:
                        tool_calls_data[idx] = {
                            "id": "",
                            "function": {"name": "", "arguments": ""},
                            "type": "function",
                        }
                    if tc.id:
                        tool_calls_data[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_data[idx]["function"]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_calls_data[idx]["function"]["arguments"] += tc.function.arguments

        # After stream is done, check what we got

        # If there were tool calls, run them
        if tool_calls_data:
            # Build the assistant message with tool calls
            tool_calls_list = [tool_calls_data[idx] for idx in sorted(tool_calls_data.keys())]
            assistant_msg = {"role": "assistant", "tool_calls": tool_calls_list}
            if full_content:
                assistant_msg["content"] = full_content
            messages.append(assistant_msg)

            # Run each tool
            for tc in tool_calls_list:
                tool_name = tc["function"]["name"]
                tool_args = json.loads(tc["function"]["arguments"])

                print(f"\n  [Tool] Calling: {tool_name}({tool_args})")

                func = tool_functions.get(tool_name)
                if func:
                    result = func(**tool_args)
                else:
                    result = f"Unknown tool: {tool_name}"

                print(f"  [Tool] Result: {result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

            # Continue the loop — LLM will now stream the final answer
            print()
            print("Assistant: ", end="")
            continue

        # If no tool calls, we're done — text was already printed
        if full_content:
            print()  # newline after streamed text

        messages.append({"role": "assistant", "content": full_content})
        return full_content


# ===========================================================================
# PART 3: MAIN — Choose basic streaming or agent streaming
# ===========================================================================

def main():
    print("=" * 55)
    print("  Streaming Agent (Step 12)")
    print("  Watch words appear in real-time!")
    print("  Type 'quit' to exit")
    print("  Type 'mode' to switch between chat/agent")
    print("=" * 55)
    print()

    mode = "agent"  # Start with agent mode (has tools)
    print(f"  Mode: AGENT (has weather + calculator tools)")
    print()
    print("  Try:")
    print("    'Tell me a short story about a robot'  (see streaming)")
    print("    'What is the weather in Tokyo?'         (tool + streaming)")
    print("    'What is 234 * 567?'                    (tool + streaming)")
    print()

    conversation = [
        {"role": "system", "content": """You are a helpful assistant.
You have access to weather and calculator tools. Use them when relevant.
Keep answers concise."""}
    ]

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "mode":
            mode = "chat" if mode == "agent" else "agent"
            label = "CHAT (no tools, just streaming)" if mode == "chat" else "AGENT (tools + streaming)"
            print(f"  Switched to: {label}\n")
            continue

        conversation.append({"role": "user", "content": user_input})

        print("Assistant: ", end="")

        try:
            if mode == "chat":
                response_text = stream_chat(conversation)
                conversation.append({"role": "assistant", "content": response_text})
            else:
                # stream_agent appends to conversation internally
                stream_agent(conversation)
            print()
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
