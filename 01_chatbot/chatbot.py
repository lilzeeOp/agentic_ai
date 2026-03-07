"""
STEP 1: LLM API Basics - Simple Chatbot
========================================

WHAT YOU'LL LEARN:
- How to connect to an LLM (Azure OpenAI) via API
- The "messages" format that ALL LLM APIs use
- How conversation history works
- What system prompts do

KEY CONCEPT: The Messages List
------------------------------
Every LLM API works with a list of messages. Each message has:
  - "role": who said it (system / user / assistant)
  - "content": what they said

Example:
  messages = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is Python?"},
      {"role": "assistant", "content": "Python is a programming language..."},
      {"role": "user", "content": "Why is it popular?"},
  ]

The LLM reads this ENTIRE list each time and generates the next assistant reply.
This is how it "remembers" the conversation - we send the full history every time!
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# 1. SETUP - Load API key and create the client
# ---------------------------------------------------------------------------

# load_dotenv() reads the .env file and puts values into environment variables
# This keeps our API key out of the code (never hardcode secrets!)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# We use the regular OpenAI client (not AzureOpenAI) because the Azure endpoint
# at /openai/v1 is OpenAI-compatible. This is simpler and works the same way.
client = OpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT") + "/openai/v1",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-5.2-chat")

# ---------------------------------------------------------------------------
# 2. THE SYSTEM PROMPT - This defines the AI's personality and rules
# ---------------------------------------------------------------------------
# The system prompt is the FIRST message. It sets behavior for the entire conversation.
# This is one of the most powerful tools in prompt engineering.

SYSTEM_PROMPT = """You are a friendly and helpful AI tutor who specializes in teaching
programming and AI concepts.

Rules:
- Explain things simply, as if teaching a beginner
- Use short examples when helpful
- If you don't know something, say so honestly
- Keep responses concise (2-3 paragraphs max)
"""

# ---------------------------------------------------------------------------
# 3. CONVERSATION HISTORY - This is the "memory" of our chatbot
# ---------------------------------------------------------------------------
# We start with just the system prompt. Every user message and AI reply
# gets appended here. The ENTIRE list is sent with each API call.

conversation_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]


def chat(user_message: str) -> str:
    """
    Send a message to the LLM and get a response.

    This function does 3 things:
    1. Adds the user's message to conversation history
    2. Sends the FULL history to the API
    3. Adds the AI's reply to history and returns it
    """

    # Step 1: Add user message to history
    conversation_history.append({"role": "user", "content": user_message})

    # Step 2: Call the API with full conversation history
    # The LLM reads ALL messages to understand context
    response = client.chat.completions.create(
        model=MODEL,
        messages=conversation_history,
        max_completion_tokens=1000,
    )

    # Step 3: Extract the reply text from the API response
    # The response object has a specific structure:
    #   response.choices[0].message.content  ->  the actual text
    assistant_message = response.choices[0].message.content

    # Step 4: Add assistant reply to history (so future calls have context)
    conversation_history.append({"role": "assistant", "content": assistant_message})

    return assistant_message


# ---------------------------------------------------------------------------
# 4. MAIN LOOP - The interactive chat interface
# ---------------------------------------------------------------------------

def main():
    print("=" * 50)
    print("  AI Tutor Chatbot (Step 1)")
    print("  Type 'quit' to exit")
    print("  Type 'history' to see the message list")
    print("=" * 50)
    print()

    while True:
        # Get user input
        user_input = input("You: ").strip()

        # Handle special commands
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "history":
            # This shows you the raw messages list - the core data structure!
            print("\n--- Conversation History (what gets sent to the API) ---")
            for i, msg in enumerate(conversation_history):
                role = msg["role"].upper()
                content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
                print(f"  [{i}] {role}: {content}")
            print("--- End History ---\n")
            continue

        # Send to LLM and print response
        try:
            response = chat(user_input)
            print(f"\nAssistant: {response}\n")
        except Exception as e:
            print(f"\nError: {e}\n")
            # Remove the failed user message from history
            conversation_history.pop()


if __name__ == "__main__":
    main()
