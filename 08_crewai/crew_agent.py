"""
STEP 8: CrewAI — Building Agent Teams the Easy Way
====================================================

WHAT YOU'LL LEARN:
- How CrewAI makes multi-agent systems simple
- The 4 building blocks: Agent, Task, Crew, Process
- How to configure a custom LLM with CrewAI
- Sequential vs Hierarchical workflows

COMPARING WITH WHAT WE BUILT BEFORE:
--------------------------------------
Step 5 (manual):    ~370 lines of code, wrote everything from scratch
Step 6 (LangGraph): ~370 lines, defined graph nodes and edges
Step 8 (CrewAI):    ~80 lines, describe agents like people

CrewAI thinks in PEOPLE terms:
- Agent = a person with a role, goal, and backstory
- Task = a piece of work assigned to a person
- Crew = a team of people working together
- Process = how the team collaborates (assembly line or manager-led)

THE CREW WE'LL BUILD:
-----------------------
  Researcher → finds facts about a topic
  Writer     → turns facts into an article
  Editor     → reviews and polishes the article

  Process: Sequential (Researcher first, then Writer, then Editor)
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


# ===========================================================================
# PART 1: CONFIGURE THE LLM
# ===========================================================================
# CrewAI needs to know which LLM to use.
# We create an LLM object pointing to our Azure endpoint.

llm = LLM(
    model="gpt-5.2-chat",
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT") + "/openai/v1",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)


# ===========================================================================
# PART 2: CREATE THE AGENTS (the team members)
# ===========================================================================
# Each agent has:
#   - role: their job title
#   - goal: what they're trying to achieve
#   - backstory: their experience (helps the LLM get into character)
#
# Think of it as writing a job posting for each team member.

researcher = Agent(
    role="Senior Research Analyst",
    goal="Find comprehensive, accurate, and up-to-date information about the given topic",
    backstory="""You are an experienced research analyst with 15 years of experience.
You are meticulous about facts, always cite specifics like numbers, dates,
and names. You organize your findings into clear, structured bullet points.
You never make up information — if you're unsure, you say so.""",
    llm=llm,
    verbose=True,  # print what the agent is doing (great for learning)
)

writer = Agent(
    role="Content Writer",
    goal="Transform research findings into engaging, beginner-friendly articles",
    backstory="""You are a skilled content writer who specializes in making complex
topics simple. You write in short paragraphs, use analogies, and always
keep your audience (beginners) in mind. You never use jargon without
explaining it first. Your articles are well-structured with clear headings.""",
    llm=llm,
    verbose=True,
)

editor = Agent(
    role="Senior Editor",
    goal="Review and polish content for clarity, accuracy, and engagement",
    backstory="""You are a senior editor with a sharp eye for detail. You check
for factual errors, improve sentence flow, fix awkward phrasing, and
ensure the content is concise. You maintain the original voice while
making the writing tighter and more impactful.""",
    llm=llm,
    verbose=True,
)


# ===========================================================================
# PART 3: CREATE THE TASKS (the work to be done)
# ===========================================================================
# Each task has:
#   - description: what needs to be done (can include {variables})
#   - expected_output: what the result should look like
#   - agent: who is responsible
#
# In sequential process, each task's output automatically becomes
# available to the next task. No manual passing needed!

def create_tasks(topic: str) -> list[Task]:
    """Create a set of tasks for a given topic."""

    research_task = Task(
        description=f"""Research the topic: '{topic}'

        Provide:
        - 7-10 key facts with specific details (numbers, dates, names)
        - Why this topic matters
        - Common misconceptions if any
        - Recent developments or trends

        Be thorough but organized. Use bullet points.""",
        expected_output="A structured list of key facts and insights about the topic, with specific details and sources where possible.",
        agent=researcher,
    )

    writing_task = Task(
        description=f"""Using the research provided, write an engaging article about '{topic}'.

        Requirements:
        - Write for beginners (assume no prior knowledge)
        - Use short paragraphs (2-3 sentences each)
        - Include section headings
        - Use analogies to explain complex ideas
        - Keep it under 500 words
        - Make it engaging and easy to read""",
        expected_output="A well-structured, beginner-friendly article with headings, short paragraphs, and clear explanations.",
        agent=writer,
    )

    editing_task = Task(
        description="""Review and polish the article.

        Check for:
        - Factual accuracy
        - Grammar and spelling
        - Sentence flow and readability
        - Remove any unnecessary jargon
        - Ensure it's truly beginner-friendly

        Output the final polished version of the article.""",
        expected_output="The final, polished version of the article ready for publication.",
        agent=editor,
    )

    return [research_task, writing_task, editing_task]


# ===========================================================================
# PART 4: CREATE THE CREW AND RUN
# ===========================================================================

def run_crew(topic: str) -> str:
    """
    Assemble the crew and kick off the work.

    Process.sequential means:
      research_task runs first
      → its output feeds into writing_task
      → its output feeds into editing_task
      → final result returned

    This is like an assembly line:
      Researcher does their part → passes to Writer → passes to Editor → Done
    """

    tasks = create_tasks(topic)

    # Assemble the crew
    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=tasks,
        process=Process.sequential,  # assembly line: one after another
        verbose=True,                # show what's happening
    )

    # Kick off the work
    result = crew.kickoff()
    return result.raw


# ===========================================================================
# PART 5: MAIN
# ===========================================================================

def main():
    print("=" * 55)
    print("  CrewAI Agent Team (Step 8)")
    print("  Researcher → Writer → Editor")
    print("  Type 'quit' to exit")
    print("=" * 55)
    print()
    print("  Give a topic and the crew will:")
    print("    1. Research it (Researcher)")
    print("    2. Write an article (Writer)")
    print("    3. Polish it (Editor)")
    print()
    print("  Try: 'What is Docker and why do developers use it'")
    print()

    while True:
        topic = input("Topic: ").strip()
        if not topic:
            continue
        if topic.lower() == "quit":
            print("Goodbye!")
            break

        try:
            print(f"\n  Crew assembled! Starting work on: {topic}\n")
            print("=" * 55)
            result = run_crew(topic)
            safe = result.encode("utf-8", errors="replace").decode("utf-8")
            print("\n" + "=" * 55)
            print("  FINAL ARTICLE")
            print("=" * 55)
            print(f"\n{safe}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
