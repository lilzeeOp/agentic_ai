"""
STEP 5: Multi-Agent Systems — A Team of Agents
================================================

WHAT YOU'LL LEARN:
- How to create multiple specialized agents
- How an orchestrator delegates tasks to the right agent
- How agents pass information between each other
- Why multi-agent is better than one big agent for complex tasks

THE ARCHITECTURE:
------------------

                    User
                      |
                 Orchestrator
                (decides who to call)
               /       |        \
         Researcher   Writer    Coder
         (finds       (writes   (writes
          facts)      content)   code)

Each agent is just an LLM call with a DIFFERENT system prompt.
That's it. An "agent" is just a system prompt + optional tools.
The orchestrator is also an agent — its "tools" are the other agents.

WHY NOT ONE BIG AGENT?
-----------------------
1. Focused prompts work better than "do everything" prompts
2. You can improve each agent independently
3. Different agents could use different models (cheap model for simple tasks)
4. Easier to debug — you know which agent went wrong
"""

import os
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
# PART 1: THE SPECIALIST AGENTS
# ===========================================================================
# Each agent is a function that calls the LLM with a specialized prompt.
# They are experts in ONE thing.

def call_llm(system_prompt: str, user_message: str, max_tokens: int = 1500) -> str:
    """Helper: call the LLM with a given system prompt and message."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_completion_tokens=max_tokens,
    )
    return response.choices[0].message.content


# --- Agent 1: Researcher ---
# Good at finding and organizing facts

def research_agent(topic: str) -> str:
    """Research agent: gathers key facts about a topic."""
    print(f"  [Researcher] Researching: {topic}")

    result = call_llm(
        system_prompt="""You are a research specialist. Your job is to provide
key facts and information about a given topic.

Rules:
- Provide 5-7 key facts as bullet points
- Include numbers, dates, and specifics when possible
- Keep it factual, no opinions
- Format as a clean bullet list""",
        user_message=f"Research this topic and give me key facts:\n\n{topic}"
    )

    print(f"  [Researcher] Done.")
    return result


# --- Agent 2: Writer ---
# Good at turning raw facts into polished content

def writer_agent(task: str) -> str:
    """Writer agent: creates polished written content."""
    print(f"  [Writer] Writing content...")

    result = call_llm(
        system_prompt="""You are a professional content writer. Your job is to take
information and turn it into well-written, engaging content.

Rules:
- Write in a clear, friendly tone suitable for beginners
- Use short paragraphs (2-3 sentences each)
- Add section headers where appropriate
- Make it engaging and easy to read
- Keep it concise (under 300 words)""",
        user_message=task,
        max_tokens=2000,
    )

    print(f"  [Writer] Done.")
    return result


# --- Agent 3: Coder ---
# Good at writing code examples

def coder_agent(task: str) -> str:
    """Coder agent: writes Python code examples."""
    print(f"  [Coder] Writing code...")

    result = call_llm(
        system_prompt="""You are an expert Python programmer. Your job is to write
clean, well-commented Python code examples.

Rules:
- Write simple, beginner-friendly code
- Add comments explaining each step
- Include a brief explanation before the code
- Use only Python standard library (no pip packages)
- Keep examples short and focused (under 30 lines each)""",
        user_message=task,
    )

    print(f"  [Coder] Done.")
    return result


# --- Agent 4: Critic ---
# Reviews and improves the final output

def critic_agent(content: str) -> str:
    """Critic agent: reviews content and suggests improvements."""
    print(f"  [Critic] Reviewing output...")

    result = call_llm(
        system_prompt="""You are a quality reviewer. Your job is to review content
and provide a brief, improved final version.

Rules:
- Fix any factual errors
- Improve clarity if needed
- Make sure it flows well
- Keep the same length and structure
- Output ONLY the improved version, no commentary""",
        user_message=f"Review and improve this content:\n\n{content}",
        max_tokens=2000,
    )

    print(f"  [Critic] Done.")
    return result


# ===========================================================================
# PART 2: THE ORCHESTRATOR
# ===========================================================================
# The orchestrator is the "boss" agent. It reads the user's request and
# decides which agents to call and in what order.
# Its "tools" are the other agents.

orchestrator_tools = [
    {
        "type": "function",
        "function": {
            "name": "research_agent",
            "description": "Call the Research Agent to gather facts about a topic. Use when you need factual information before writing or answering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to research"
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "writer_agent",
            "description": "Call the Writer Agent to create polished written content. Pass it the topic AND any research results so it has material to work with.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "What to write, including any context or research to use"
                    }
                },
                "required": ["task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "coder_agent",
            "description": "Call the Coder Agent to write Python code examples. Use when the task involves programming or code demonstrations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "What code to write and any context"
                    }
                },
                "required": ["task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "critic_agent",
            "description": "Call the Critic Agent to review and improve content. Use as a final step to polish the output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to review and improve"
                    }
                },
                "required": ["content"]
            }
        }
    }
]

agent_functions = {
    "research_agent": research_agent,
    "writer_agent": writer_agent,
    "coder_agent": coder_agent,
    "critic_agent": critic_agent,
}


ORCHESTRATOR_PROMPT = """You are an orchestrator that manages a team of specialist agents.
Your job is to break down the user's request and delegate to the right agents.

Your team:
1. Research Agent — finds facts about any topic
2. Writer Agent — writes polished articles, explanations, blog posts
3. Coder Agent — writes Python code examples
4. Critic Agent — reviews and improves content

Workflow for a typical request:
1. First, call research_agent to gather facts
2. Then, call writer_agent with the research results to create content
3. If code is needed, call coder_agent
4. Optionally, call critic_agent to review the final output
5. Combine everything into a final response

IMPORTANT:
- Always research before writing (give the writer actual material to work with)
- Pass the research results TO the writer in the task description
- For simple questions, you can answer directly without agents
- When combining agent outputs, format them nicely with headers
"""


def run_orchestrator(user_message: str) -> str:
    """
    The orchestrator loop. Same as the agent loop in Step 2,
    but now the "tools" are other agents instead of simple functions.
    """
    messages = [
        {"role": "system", "content": ORCHESTRATOR_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # Collect all agent outputs to display at the end
    agent_outputs = {}
    step_count = 0

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=orchestrator_tools,
            max_completion_tokens=2000,
        )

        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                step_count += 1

                print(f"\n  Step {step_count}: Calling {tool_name}")
                print(f"  {'─' * 45}")

                # Call the specialist agent
                func = agent_functions[tool_name]
                result = func(**tool_args)

                agent_outputs[tool_name] = result

                # Send result back to orchestrator
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            continue

        # Orchestrator is done — return final combined response
        return message.content


# ===========================================================================
# PART 3: MAIN
# ===========================================================================

def main():
    print("=" * 55)
    print("  Multi-Agent System (Step 5)")
    print("  Orchestrator + Researcher + Writer + Coder + Critic")
    print("  Type 'quit' to exit")
    print("=" * 55)
    print()
    print("  Try these:")
    print("    'Write a blog post about why Python is great'")
    print("    'Explain recursion with code examples'")
    print("    'Compare lists and tuples in Python'")
    print()

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        try:
            print(f"\n  Orchestrator received task. Delegating to team...\n")
            response = run_orchestrator(user_input)
            safe = response.encode("utf-8", errors="replace").decode("utf-8")
            print(f"\n{'='*55}")
            print("  FINAL OUTPUT")
            print(f"{'='*55}")
            print(f"\n{safe}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
