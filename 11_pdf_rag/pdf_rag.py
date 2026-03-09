"""
STEP 11: PDF RAG — Chat With Your PDF Files
=============================================

WHAT YOU'LL LEARN:
- How to extract text from PDF files
- How to chunk PDF content intelligently (by pages and paragraphs)
- How to build a RAG pipeline over PDF documents
- How to cite page numbers in answers

HOW IT WORKS:
--------------
Same RAG pipeline as Step 7, but now with real PDFs:

1. LOAD   → Read PDFs using PyMuPDF (extracts text from each page)
2. CHUNK  → Split into smaller pieces (paragraph-level)
3. EMBED  → Convert chunks to TF-IDF vectors (free, local)
4. SEARCH → Find relevant chunks for a question
5. ANSWER → Send chunks + question to LLM

NEW COMPARED TO STEP 7:
- Reads actual PDF files (not just .txt)
- Tracks page numbers (so answers cite "Page 3 of handbook.pdf")
- Handles multiple PDFs at once
- Better chunking (respects page boundaries)

You can drop ANY PDF into the pdfs/ folder and chat with it.
"""

import os
import glob
import pymupdf
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

client = OpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT") + "/openai/v1",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)
MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-5.2-chat")

PDF_DIR = os.path.join(os.path.dirname(__file__), "pdfs")


# ===========================================================================
# PART 1: PDF TEXT EXTRACTION
# ===========================================================================

def extract_text_from_pdf(filepath: str) -> list[dict]:
    """
    Extract text from a PDF file, page by page.

    Returns a list of dicts:
      [{"page": 1, "text": "...", "filename": "handbook.pdf"}, ...]

    PyMuPDF reads each page and gives us the raw text.
    We keep track of page numbers for citation.
    """
    filename = os.path.basename(filepath)
    pages = []

    doc = pymupdf.open(filepath)
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():  # skip empty pages
            pages.append({
                "page": page_num + 1,
                "text": text.strip(),
                "filename": filename,
            })
    doc.close()

    return pages


def load_all_pdfs(directory: str) -> list[dict]:
    """Load all PDF files from a directory."""
    all_pages = []
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))

    if not pdf_files:
        print(f"  No PDF files found in {directory}")
        return []

    for filepath in pdf_files:
        filename = os.path.basename(filepath)
        pages = extract_text_from_pdf(filepath)
        all_pages.extend(pages)
        print(f"  Loaded: {filename} ({len(pages)} pages)")

    return all_pages


# ===========================================================================
# PART 2: CHUNKING
# ===========================================================================

def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Split PDF pages into smaller chunks.

    Each page might have multiple topics. We split by paragraphs
    (double newlines) but keep chunks that are at least 80 chars.
    Each chunk remembers which file and page it came from.
    """
    chunks = []
    for page in pages:
        # Split page text into paragraphs
        paragraphs = page["text"].split("\n\n")

        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            current_chunk += para + "\n\n"

            # If chunk is big enough, save it
            if len(current_chunk) >= 200:
                chunks.append({
                    "text": current_chunk.strip(),
                    "filename": page["filename"],
                    "page": page["page"],
                })
                current_chunk = ""

        # Save any remaining text
        if len(current_chunk.strip()) >= 80:
            chunks.append({
                "text": current_chunk.strip(),
                "filename": page["filename"],
                "page": page["page"],
            })

    return chunks


# ===========================================================================
# PART 3: VECTOR STORE (same as Step 7)
# ===========================================================================

class PDFVectorStore:
    """Vector store for PDF chunks using TF-IDF."""

    def __init__(self):
        self.chunks = []
        self.vectorizer = None
        self.vectors = None

    def add_chunks(self, chunks: list[dict]):
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
        )
        self.vectors = self.vectorizer.fit_transform(texts)
        print(f"  Indexed {len(chunks)} chunks into {self.vectors.shape[1]}-dimensional vectors")

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        """Find the most relevant chunks for a query."""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors)[0]

        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.0:
                results.append({
                    "text": self.chunks[idx]["text"],
                    "filename": self.chunks[idx]["filename"],
                    "page": self.chunks[idx]["page"],
                    "score": round(float(similarities[idx]), 3),
                })
        return results


# ===========================================================================
# PART 4: RAG ANSWER
# ===========================================================================

def rag_answer(query: str, vector_store: PDFVectorStore) -> str:
    """Search PDFs and answer using retrieved context."""

    print(f"\n  [PDF RAG] Searching...")
    results = vector_store.search(query, top_k=4)

    if not results:
        return "No relevant information found in the PDFs."

    print(f"  [PDF RAG] Found {len(results)} relevant chunks:")
    for i, r in enumerate(results):
        preview = r["text"][:70].replace("\n", " ") + "..."
        print(f"    {i+1}. [{r['filename']} p.{r['page']}] (score: {r['score']}) {preview}")

    # Build context with source citations
    context = "\n\n---\n\n".join(
        f"[Source: {r['filename']}, Page {r['page']}]\n{r['text']}" for r in results
    )

    print(f"  [PDF RAG] Asking LLM...")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": """You are a helpful assistant that answers questions
based on PDF documents.

RULES:
- ONLY use information from the provided context
- Always cite the source file and page number (e.g., "According to employee_handbook.pdf, page 2...")
- If the context doesn't contain the answer, say "I couldn't find that in the documents"
- Be specific and quote relevant details
- Keep answers concise"""},
            {"role": "user", "content": f"""Context from PDF documents:

{context}

---

Question: {query}"""},
        ],
        max_completion_tokens=1000,
    )

    return response.choices[0].message.content


# ===========================================================================
# PART 5: MAIN
# ===========================================================================

def main():
    print("=" * 55)
    print("  PDF RAG Agent (Step 11)")
    print("  Chat with your PDF files!")
    print("  Type 'quit' to exit")
    print("  Type 'docs' to see loaded documents")
    print("=" * 55)
    print()

    # Load and index PDFs
    print("  Loading PDFs...")
    pages = load_all_pdfs(PDF_DIR)

    if not pages:
        print("  No PDFs found. Put PDF files in the pdfs/ folder.")
        print("  Run 'python create_sample_pdf.py' to create samples.")
        return

    chunks = chunk_pages(pages)
    print(f"  Created {len(chunks)} chunks from {len(pages)} pages")

    vector_store = PDFVectorStore()
    vector_store.add_chunks(chunks)
    print()

    # Show loaded files
    filenames = sorted(set(p["filename"] for p in pages))
    print("  Loaded PDFs:")
    for f in filenames:
        page_count = len([p for p in pages if p["filename"] == f])
        print(f"    - {f} ({page_count} pages)")
    print()

    print("  Try asking:")
    print("    'How many days of annual leave do employees get?'")
    print("    'What is the API rate limit?'")
    print("    'How do I authenticate with the API?'")
    print("    'What is the remote work policy?'")
    print()

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "docs":
            for f in filenames:
                page_count = len([p for p in pages if p["filename"] == f])
                print(f"  - {f} ({page_count} pages)")
            print()
            continue

        try:
            answer = rag_answer(user_input, vector_store)
            safe = answer.encode("utf-8", errors="replace").decode("utf-8")
            print(f"\nAssistant: {safe}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
