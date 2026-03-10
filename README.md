# Agentic AI — Learn by Building

A step-by-step journey from zero to building AI agents. Each folder is one concept, explained in plain English with working code.

## Prerequisites

- Python (can read and understand it)
- An LLM API key (this project uses Azure OpenAI)

## Setup

```bash
# Clone the repo
git clone https://github.com/lilzeeOp/agentic_ai.git
cd agentic_ai

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install openai python-dotenv scikit-learn requests beautifulsoup4 pymupdf mcp langchain-openai langgraph crewai fastapi uvicorn

# Create .env file with your API credentials
# AZURE_OPENAI_API_KEY=your_key
# AZURE_OPENAI_ENDPOINT=your_endpoint
# AZURE_OPENAI_MODEL=your_model
```

## The Steps

| Step | Folder | What You Learn | Concept Level |
|------|--------|---------------|---------------|
| 1 | `01_chatbot/` | Talk to an LLM via API | Gen AI |
| 2 | `02_tool_calling/` | LLM decides to use tools (calculator, time) | AI Agent |
| 3 | `03_react_agent/` | Thought-Action-Observation reasoning loop | AI Agent |
| 4 | `04_memory_agent/` | Short-term + long-term memory, summarization | AI Agent |
| 5 | `05_multi_agent/` | Multiple specialist agents + orchestrator | Agentic AI |
| 6 | `06_langgraph/` | Graph-based agent with conditional routing | Agentic AI |
| 7 | `07_rag/` | Search your own documents with RAG + TF-IDF | Agentic AI |
| 8 | `08_crewai/` | CrewAI framework — agents as a team | Agentic AI |
| 9 | `09_mcp/` | MCP server + client — universal tool protocol | Agentic AI |
| 10 | `10_web_agent/` | Agent that browses the web autonomously | Agentic AI |
| 11 | `11_pdf_rag/` | Chat with PDF files, cite page numbers | Agentic AI |
| 12 | `12_streaming/` | Real-time token-by-token output (typing effect) | Agentic AI |
| 13 | `13_fastapi/` | Turn your agent into a web API (WIP) | Agentic AI |

## How to Run Each Step

```bash
# Activate venv first, then:
cd 01_chatbot && python chatbot.py
cd 02_tool_calling && python agent.py
cd 03_react_agent && python react_agent.py
cd 04_memory_agent && python memory_agent.py
cd 05_multi_agent && python multi_agent.py
cd 06_langgraph && python langgraph_agent.py
cd 07_rag && python rag_agent.py
cd 08_crewai && python crew_agent.py
cd 09_mcp && python mcp_client.py
cd 10_web_agent && python web_agent.py
cd 11_pdf_rag && python create_sample_pdf.py && python pdf_rag.py
cd 12_streaming && python streaming_chat.py
cd 13_fastapi && uvicorn api_agent:app --reload
```

## Key Concepts

**Gen AI** — LLM answers questions. One input, one output. (Step 1)

**AI Agent** — LLM + tools. It can take actions, not just talk. (Steps 2-4)

**Agentic AI** — Autonomous, multi-step, self-correcting, often multi-agent. Plans, executes, recovers from errors. (Steps 5-13)

## Full Story

Read `THE_COMPLETE_STORY.md` for the complete history and explanation of every concept in plain English.
