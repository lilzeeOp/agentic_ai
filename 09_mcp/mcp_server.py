"""
STEP 9 (Part 1): MCP Server — Building a Tool Server
======================================================

WHAT YOU'LL LEARN:
- How to create an MCP server that exposes tools
- How any AI agent can discover and use your tools
- The FastMCP decorator pattern
- How stdio transport works

WHAT IS AN MCP SERVER?
-----------------------
An MCP server is a program that HOSTS tools. It says:
  "Hey, I have these tools available. Here's what they do.
   Send me a request and I'll run the tool for you."

Any MCP-compatible client (Claude, your custom agent, etc.)
can connect to this server and use the tools — without writing
custom integration code.

Think of it like a restaurant:
  - The SERVER is the restaurant (has the kitchen and menu)
  - The CLIENT is the customer (reads the menu and orders)
  - The TOOLS are the dishes (what the restaurant can make)

HOW IT COMMUNICATES:
---------------------
We use "stdio" transport — the client starts this server as a
subprocess and communicates through stdin/stdout (like piping).

  Client starts server → sends requests via stdin → reads responses from stdout

This is the simplest transport. Other options: HTTP (SSE), WebSocket.
"""

import os
import json
from datetime import datetime
from mcp.server.fastmcp import FastMCP

# Create the MCP server
# The name identifies this server to clients
server = FastMCP(
    "learning-tools",
    instructions="A collection of learning tools: notes, calculator, file search, and timer."
)

# Path for persistent notes storage
NOTES_FILE = os.path.join(os.path.dirname(__file__), "notes.json")


# ===========================================================================
# TOOL 1: Notes Manager
# ===========================================================================
# A simple note-taking tool. The agent can save and retrieve notes.
# Each tool is just a Python function with a @server.tool() decorator.
# MCP reads the function name, docstring, and type hints to build
# the tool description automatically.

def _load_notes() -> list[dict]:
    """Load notes from disk."""
    if os.path.exists(NOTES_FILE):
        with open(NOTES_FILE, "r") as f:
            return json.load(f)
    return []


def _save_notes(notes: list[dict]):
    """Save notes to disk."""
    with open(NOTES_FILE, "w") as f:
        json.dump(notes, f, indent=2)


@server.tool()
def save_note(title: str, content: str) -> str:
    """Save a note with a title and content. Use this to remember information for later."""
    notes = _load_notes()
    note = {
        "title": title,
        "content": content,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    notes.append(note)
    _save_notes(notes)
    return f"Note saved: '{title}'"


@server.tool()
def list_notes() -> str:
    """List all saved notes. Returns titles and creation dates."""
    notes = _load_notes()
    if not notes:
        return "No notes saved yet."
    result = []
    for i, note in enumerate(notes, 1):
        result.append(f"{i}. [{note['created_at']}] {note['title']}")
    return "\n".join(result)


@server.tool()
def read_note(title: str) -> str:
    """Read a specific note by its title."""
    notes = _load_notes()
    for note in notes:
        if note["title"].lower() == title.lower():
            return f"Title: {note['title']}\nCreated: {note['created_at']}\n\n{note['content']}"
    return f"Note '{title}' not found."


@server.tool()
def delete_note(title: str) -> str:
    """Delete a note by its title."""
    notes = _load_notes()
    filtered = [n for n in notes if n["title"].lower() != title.lower()]
    if len(filtered) == len(notes):
        return f"Note '{title}' not found."
    _save_notes(filtered)
    return f"Note '{title}' deleted."


# ===========================================================================
# TOOL 2: Calculator
# ===========================================================================

@server.tool()
def calculator(expression: str) -> str:
    """Evaluate a math expression. Supports +, -, *, /, parentheses, and decimals.
    Example: '(10 + 5) * 3 / 2'"""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return f"Error: Expression contains invalid characters."
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# ===========================================================================
# TOOL 3: File Reader
# ===========================================================================

@server.tool()
def read_file(filepath: str) -> str:
    """Read the contents of a text file. Provide the full file path."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) > 5000:
            content = content[:5000] + "\n... (truncated, file too long)"
        return content
    except FileNotFoundError:
        return f"File not found: {filepath}"
    except Exception as e:
        return f"Error reading file: {e}"


@server.tool()
def list_files(directory: str) -> str:
    """List files in a directory. Provide the full directory path."""
    try:
        entries = os.listdir(directory)
        if not entries:
            return f"Directory '{directory}' is empty."
        result = []
        for entry in sorted(entries):
            full_path = os.path.join(directory, entry)
            if os.path.isdir(full_path):
                result.append(f"[DIR]  {entry}")
            else:
                size = os.path.getsize(full_path)
                result.append(f"[FILE] {entry} ({size} bytes)")
        return "\n".join(result)
    except FileNotFoundError:
        return f"Directory not found: {directory}"
    except Exception as e:
        return f"Error: {e}"


# ===========================================================================
# TOOL 4: Date/Time
# ===========================================================================

@server.tool()
def get_current_datetime() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S (%A)")


@server.tool()
def days_between(date1: str, date2: str) -> str:
    """Calculate the number of days between two dates. Format: YYYY-MM-DD"""
    try:
        d1 = datetime.strptime(date1, "%Y-%m-%d")
        d2 = datetime.strptime(date2, "%Y-%m-%d")
        diff = abs((d2 - d1).days)
        return f"{diff} days between {date1} and {date2}"
    except ValueError:
        return "Error: Use format YYYY-MM-DD (e.g., 2024-01-15)"


# ===========================================================================
# RUN THE SERVER
# ===========================================================================
# When this script is executed, it starts the MCP server using stdio transport.
# The client will start this as a subprocess and communicate through stdin/stdout.

if __name__ == "__main__":
    server.run("stdio")
