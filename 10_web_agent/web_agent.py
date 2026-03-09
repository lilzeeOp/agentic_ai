"""
STEP 10: Web Browsing Agent
=============================

WHAT YOU'LL LEARN:
- How to fetch and parse real web pages
- How to extract clean text from HTML
- How to build an agent that browses the web autonomously
- How to handle real-world web scraping challenges

HOW IT WORKS:
--------------
1. User asks a question (e.g., "What's on the Python.org homepage?")
2. LLM decides to use the browse_webpage tool with a URL
3. Our code fetches the page, strips HTML, extracts clean text
4. LLM reads the extracted text and answers the question

TOOLS:
-------
- browse_webpage(url)       → fetch a URL and extract its text content
- search_web(query)         → search DuckDuckGo and return result links
- extract_links(url)        → get all links from a page
- extract_headlines(url)    → get all headings (h1, h2, h3) from a page

This is how Perplexity, ChatGPT Browse, and other AI search tools work
at a basic level: fetch pages → extract text → send to LLM.
"""

import os
import re
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urljoin
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

client = OpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT") + "/openai/v1",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)
MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-5.2-chat")

# Pretend to be a browser so websites don't block us
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


# ===========================================================================
# PART 1: WEB TOOLS
# ===========================================================================

def browse_webpage(url: str) -> str:
    """
    Fetch a webpage and extract its text content.

    This is the core of web browsing:
    1. Download the raw HTML (like viewing page source)
    2. Parse it with BeautifulSoup (understands HTML structure)
    3. Remove scripts, styles, and other junk
    4. Extract just the readable text
    """
    try:
        print(f"  [Web] Fetching: {url}")
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove non-content elements (scripts, styles, navbars, footers)
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]):
            tag.decompose()

        # Extract text
        text = soup.get_text(separator="\n", strip=True)

        # Clean up: remove excessive blank lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

        # Truncate if too long (LLM context limits)
        if len(text) > 6000:
            text = text[:6000] + "\n\n... [content truncated — page too long]"

        print(f"  [Web] Extracted {len(text)} chars of text")
        return text

    except requests.exceptions.Timeout:
        return f"Error: Request timed out for {url}"
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP {e.response.status_code} for {url}"
    except Exception as e:
        return f"Error fetching {url}: {e}"


def search_web(query: str) -> str:
    """
    Search DuckDuckGo and return results.

    We use DuckDuckGo's HTML page (no API key needed, free).
    Parse the results to get titles, URLs, and snippets.
    """
    try:
        print(f"  [Web] Searching: {query}")
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        response = requests.get(search_url, headers=HEADERS, timeout=15)

        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        # DuckDuckGo HTML results are in divs with class "result"
        for i, result_div in enumerate(soup.select(".result"), 1):
            if i > 5:  # top 5 results
                break

            title_tag = result_div.select_one(".result__title a, .result__a")
            snippet_tag = result_div.select_one(".result__snippet")

            title = title_tag.get_text(strip=True) if title_tag else "No title"
            link = title_tag.get("href", "") if title_tag else ""
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else "No description"

            # DuckDuckGo wraps links in a redirect URL, extract the real one
            if "uddg=" in link:
                from urllib.parse import unquote, parse_qs, urlparse
                parsed = urlparse(link)
                params = parse_qs(parsed.query)
                if "uddg" in params:
                    link = unquote(params["uddg"][0])

            results.append(f"{i}. {title}\n   URL: {link}\n   {snippet}")

        if not results:
            return f"No results found for '{query}'"

        print(f"  [Web] Found {len(results)} results")
        return "\n\n".join(results)

    except Exception as e:
        return f"Error searching: {e}"


def extract_links(url: str) -> str:
    """Extract all links from a webpage."""
    try:
        print(f"  [Web] Extracting links from: {url}")
        response = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")

        links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            text = a_tag.get_text(strip=True)
            if not text or len(text) < 3:
                continue
            # Make relative URLs absolute
            full_url = urljoin(url, href)
            if full_url.startswith("http"):
                links.append(f"- {text}: {full_url}")

        if not links:
            return "No links found on this page."

        # Limit to 20 links
        links = links[:20]
        return "\n".join(links)

    except Exception as e:
        return f"Error: {e}"


def extract_headlines(url: str) -> str:
    """Extract all headings (h1, h2, h3) from a webpage. Useful for understanding page structure."""
    try:
        print(f"  [Web] Extracting headlines from: {url}")
        response = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")

        headlines = []
        for tag in soup.find_all(["h1", "h2", "h3"]):
            text = tag.get_text(strip=True)
            if text:
                level = tag.name.upper()
                headlines.append(f"[{level}] {text}")

        if not headlines:
            return "No headlines found."

        return "\n".join(headlines)

    except Exception as e:
        return f"Error: {e}"


# ===========================================================================
# PART 2: TOOL DEFINITIONS FOR THE LLM
# ===========================================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web using DuckDuckGo. Returns top 5 results with titles, URLs, and snippets. Use this when you need to find information or URLs about a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browse_webpage",
            "description": "Fetch a webpage and extract its text content. Use this to read the actual content of a URL. Provide the full URL including https://.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The full URL to fetch (e.g., https://example.com)"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_links",
            "description": "Get all links from a webpage. Useful for finding related pages, documentation links, or navigation structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to extract links from"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_headlines",
            "description": "Get all headings (h1, h2, h3) from a webpage. Useful for quickly understanding what a page covers without reading everything.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to extract headlines from"}
                },
                "required": ["url"]
            }
        }
    }
]

tool_functions = {
    "search_web": search_web,
    "browse_webpage": browse_webpage,
    "extract_links": extract_links,
    "extract_headlines": extract_headlines,
}


# ===========================================================================
# PART 3: THE AGENT LOOP
# ===========================================================================

SYSTEM_PROMPT = """You are a web research agent that can browse the internet.

You have 4 tools:
1. search_web — search DuckDuckGo for information
2. browse_webpage — read the full content of a URL
3. extract_links — get all links from a page
4. extract_headlines — get headings from a page

Workflow for answering questions:
1. If you need to find something, SEARCH first to get URLs
2. Then BROWSE the most relevant URL to read the actual content
3. Answer based on what you actually read (not your training data)

Rules:
- Always cite the URL where you found the information
- If search results are enough to answer, you don't need to browse
- If you browse a page and it doesn't have what you need, try another URL
- Keep answers concise and factual
"""

conversation = [{"role": "system", "content": SYSTEM_PROMPT}]


def run_agent(user_message: str) -> str:
    """Agent loop with web tools."""
    conversation.append({"role": "user", "content": user_message})

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=conversation,
            tools=tools,
            max_completion_tokens=1500,
        )

        message = response.choices[0].message

        if message.tool_calls:
            conversation.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                func = tool_functions.get(tool_name)
                if func:
                    result = func(**tool_args)
                else:
                    result = f"Unknown tool: {tool_name}"

                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
            continue

        conversation.append({"role": "assistant", "content": message.content})
        return message.content


# ===========================================================================
# PART 4: MAIN
# ===========================================================================

def main():
    print("=" * 55)
    print("  Web Browsing Agent (Step 10)")
    print("  An agent that can search and read the internet!")
    print("  Type 'quit' to exit")
    print("=" * 55)
    print()
    print("  Try:")
    print("    'Search for latest Python news'")
    print("    'What is on the homepage of python.org?'")
    print("    'Find and summarize the Wikipedia page about Nikola Tesla'")
    print("    'What are the top stories on Hacker News right now?'")
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
            safe = response.encode("utf-8", errors="replace").decode("utf-8")
            print(f"\nAssistant: {safe}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
