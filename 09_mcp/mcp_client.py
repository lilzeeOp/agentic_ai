"""
STEP 9 (Part 2): MCP Client — Connecting an Agent to an MCP Server
====================================================================

WHAT YOU'LL LEARN:
- How to connect to an MCP server
- How to discover tools automatically (no hardcoding!)
- How to wire MCP tools into an LLM for tool calling
- The full flow: Client → Server → Tool → Result → LLM

THE FLOW:
----------
1. Client starts the MCP server as a subprocess
2. Client asks: "What tools do you have?"
3. Server responds with tool list (names, descriptions, parameters)
4. Client converts these into OpenAI tool format
5. User asks a question → LLM decides which tool to call
6. Client sends the tool call to the MCP server
7. Server runs the tool and returns the result
8. Client sends result back to LLM → LLM gives final answer

THE KEY INSIGHT:
-----------------
The client NEVER hardcodes tool definitions. It discovers them
from the server at runtime. This means:
- Swap the server → get different tools, zero code changes
- Add tools to the server → client sees them automatically
- Any MCP-compatible server works with this client
"""

import os
import sys
import json
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# LLM client (only used for chat, everything else is MCP)
llm_client = OpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT") + "/openai/v1",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)
MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-5.2-chat")

# Path to our MCP server
SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "mcp_server.py")


def mcp_tools_to_openai_format(mcp_tools) -> list[dict]:
    """
    Convert MCP tool definitions to OpenAI's tool format.

    MCP and OpenAI use slightly different JSON formats for tools.
    This function translates between them so the LLM understands
    the tools discovered from the MCP server.
    """
    openai_tools = []
    for tool in mcp_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}},
            }
        })
    return openai_tools


async def run_agent():
    """
    Main agent loop:
    1. Connect to MCP server
    2. Discover tools
    3. Chat loop: user asks → LLM calls tools via MCP → answer
    """

    # ------------------------------------------------------------------
    # Step 1: Connect to the MCP server
    # ------------------------------------------------------------------
    # We start the server as a subprocess. The client and server
    # communicate through stdin/stdout (stdio transport).

    print("  Connecting to MCP server...")

    # Find the Python executable in our venv
    python_path = sys.executable

    server_params = StdioServerParameters(
        command=python_path,
        args=[SERVER_SCRIPT],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection (required first step)
            await session.initialize()
            print("  Connected!")

            # ----------------------------------------------------------
            # Step 2: Discover tools from the server
            # ----------------------------------------------------------
            # This is the magic of MCP — we ASK the server what tools
            # it has instead of hardcoding them.

            tools_result = await session.list_tools()
            mcp_tools = tools_result.tools

            print(f"  Discovered {len(mcp_tools)} tools from server:")
            for tool in mcp_tools:
                print(f"    - {tool.name}: {tool.description[:60]}...")

            # Convert to OpenAI format for the LLM
            openai_tools = mcp_tools_to_openai_format(mcp_tools)

            # ----------------------------------------------------------
            # Step 3: Chat loop
            # ----------------------------------------------------------

            print()
            print("=" * 55)
            print("  MCP Agent (Step 9)")
            print("  Tools discovered automatically from MCP server!")
            print("  Type 'quit' to exit")
            print("=" * 55)
            print()
            print("  Try:")
            print("    'Save a note titled groceries with content: milk, eggs, bread'")
            print("    'What is 1547 * 382?'")
            print("    'What day is it today?'")
            print("    'List my notes'")
            print()

            conversation = [
                {"role": "system", "content": """You are a helpful assistant with access to tools.
Use tools when they would help answer the user's question.
For math, always use the calculator tool.
When the user asks to save or read notes, use the note tools."""}
            ]

            while True:
                try:
                    user_input = input("You: ").strip()
                except EOFError:
                    break

                if not user_input:
                    continue
                if user_input.lower() == "quit":
                    print("Goodbye!")
                    break

                conversation.append({"role": "user", "content": user_input})

                # Agent loop: keep calling tools until LLM gives a text response
                while True:
                    response = llm_client.chat.completions.create(
                        model=MODEL,
                        messages=conversation,
                        tools=openai_tools,
                        max_completion_tokens=1000,
                    )

                    message = response.choices[0].message

                    if message.tool_calls:
                        conversation.append(message)

                        for tool_call in message.tool_calls:
                            tool_name = tool_call.function.name
                            tool_args = json.loads(tool_call.function.arguments)

                            print(f"  [MCP] Calling: {tool_name}({tool_args})")

                            # Call the tool via MCP (server runs it)
                            result = await session.call_tool(
                                name=tool_name,
                                arguments=tool_args,
                            )

                            # Extract text from result
                            result_text = ""
                            for block in result.content:
                                if hasattr(block, "text"):
                                    result_text += block.text

                            print(f"  [MCP] Result: {result_text[:100]}{'...' if len(result_text) > 100 else ''}")

                            conversation.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result_text,
                            })
                        continue

                    # No tool calls — LLM gave a text response
                    conversation.append({"role": "assistant", "content": message.content})
                    safe = message.content.encode("utf-8", errors="replace").decode("utf-8")
                    print(f"\nAssistant: {safe}\n")
                    break


def main():
    asyncio.run(run_agent())


if __name__ == "__main__":
    main()
