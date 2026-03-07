"""
STEP 7: RAG — Retrieval Augmented Generation
==============================================

WHAT YOU'LL LEARN:
- How to load and chunk documents
- How embeddings work (converting text to numbers)
- How similarity search finds relevant chunks
- How to feed retrieved context to the LLM

THE RAG PIPELINE:
------------------

  [Your Documents]
        |
    1. LOAD (read text files)
        |
    2. CHUNK (split into paragraphs)
        |
    3. EMBED (convert text → numbers using TF-IDF)
        |
    4. STORE (keep in memory)
        |
   --- at query time ---
        |
    5. SEARCH (find chunks similar to the question)
        |
    6. PROMPT (send question + relevant chunks to LLM)
        |
    7. ANSWER (LLM answers using the retrieved context)


WHAT IS TF-IDF? (our embedding method)
----------------------------------------
TF-IDF = Term Frequency - Inverse Document Frequency

It converts text into numbers based on word importance:
- Words that appear often in ONE chunk but rarely in others get HIGH scores
- Common words like "the", "is", "a" get LOW scores

Example:
  Chunk about Python:  "python" gets a high score
  Chunk about ML:      "neural" gets a high score
  Both chunks:         "the" gets a low score (too common)

This is a simpler version of what OpenAI embeddings do.
OpenAI uses neural networks (smarter), we use word statistics (simpler).
The CONCEPT is identical: text → numbers → find similar ones.

NOTE: We only use the Azure chat model for the final answer.
      Embeddings are done 100% locally using scikit-learn (free).
"""

import os
import glob
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# Only the chat model uses Azure — everything else is local
client = OpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT") + "/openai/v1",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)
MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-5.2-chat")

DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")


# ===========================================================================
# PART 1: LOAD DOCUMENTS
# ===========================================================================
# Read all .txt files from the documents folder.

def load_documents(directory: str) -> list[dict]:
    """
    Load all text files from a directory.

    Returns a list of dicts:
      [{"filename": "python_basics.txt", "content": "Python was created..."}]
    """
    documents = []
    for filepath in glob.glob(os.path.join(directory, "*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        documents.append({
            "filename": os.path.basename(filepath),
            "content": content,
        })
    print(f"  Loaded {len(documents)} documents")
    return documents


# ===========================================================================
# PART 2: CHUNK DOCUMENTS
# ===========================================================================
# Split documents into smaller pieces (paragraphs).
#
# WHY CHUNK?
# - LLMs have limited context windows
# - Smaller chunks = more precise search results
# - A whole document might have 10 topics; a chunk has 1-2

def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Split documents into paragraph-sized chunks.

    Each paragraph becomes its own searchable unit.
    We keep track of which file each chunk came from.
    """
    chunks = []
    for doc in documents:
        # Split on double newlines (paragraph breaks)
        paragraphs = doc["content"].split("\n\n")
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) < 50:  # skip very short chunks (headers, etc.)
                continue
            chunks.append({
                "text": para,
                "source": doc["filename"],
                "chunk_id": f"{doc['filename']}_{i}",
            })
    print(f"  Created {len(chunks)} chunks")
    return chunks


# ===========================================================================
# PART 3: BUILD THE VECTOR STORE (embed + store)
# ===========================================================================
# Convert all chunks into TF-IDF vectors and store them.
# This is our "vector database" — just in memory.

class SimpleVectorStore:
    """
    A simple vector store using TF-IDF.

    Real vector databases (Pinecone, ChromaDB, Weaviate) do the same thing
    but with neural embeddings and disk storage. The concept is identical:
        text → numbers → store → search by similarity
    """

    def __init__(self):
        self.chunks = []            # the original text chunks
        self.vectorizer = None      # the TF-IDF model
        self.vectors = None         # the number representations

    def add_chunks(self, chunks: list[dict]):
        """Convert chunks to TF-IDF vectors and store them."""
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]

        # TfidfVectorizer does two things:
        # 1. Learns vocabulary from all chunks (which words exist)
        # 2. Converts each chunk into a vector of word importance scores
        self.vectorizer = TfidfVectorizer(
            stop_words="english",  # ignore common words like "the", "is"
            max_features=5000,     # use top 5000 words only
        )
        self.vectors = self.vectorizer.fit_transform(texts)
        print(f"  Embedded {len(chunks)} chunks into {self.vectors.shape[1]}-dimensional vectors")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Find the most similar chunks to a query.

        Steps:
        1. Convert the query into a TF-IDF vector (same method as chunks)
        2. Calculate cosine similarity between query and ALL chunks
        3. Return the top_k most similar chunks

        COSINE SIMILARITY: measures how "close" two vectors are
        - 1.0 = identical meaning
        - 0.0 = completely unrelated
        """
        # Convert query to the same vector space
        query_vector = self.vectorizer.transform([query])

        # Calculate similarity between query and every chunk
        similarities = cosine_similarity(query_vector, self.vectors)[0]

        # Get indices of top_k most similar chunks
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.0:  # only include actual matches
                results.append({
                    "text": self.chunks[idx]["text"],
                    "source": self.chunks[idx]["source"],
                    "score": round(float(similarities[idx]), 3),
                })
        return results


# ===========================================================================
# PART 4: THE RAG FUNCTION
# ===========================================================================

def rag_answer(query: str, vector_store: SimpleVectorStore) -> str:
    """
    The complete RAG pipeline:
    1. Search for relevant chunks
    2. Build a prompt with the chunks as context
    3. Send to LLM for the final answer
    """

    # Step 1: Search (local, free — no API call)
    print(f"\n  [RAG] Searching for relevant chunks...")
    results = vector_store.search(query, top_k=3)

    if not results:
        return "I couldn't find any relevant information in the documents."

    # Show what was retrieved
    print(f"  [RAG] Found {len(results)} relevant chunks:")
    for i, r in enumerate(results):
        preview = r["text"][:80] + "..." if len(r["text"]) > 80 else r["text"]
        print(f"    {i+1}. [{r['source']}] (score: {r['score']}) {preview}")

    # Step 2: Build the context from retrieved chunks
    context = "\n\n---\n\n".join(
        f"[Source: {r['source']}]\n{r['text']}" for r in results
    )

    # Step 3: Send to LLM with the retrieved context
    # This is the only part that uses the Azure API
    print(f"  [RAG] Sending to LLM with context...")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": """You are a helpful assistant that answers questions
based on the provided context.

RULES:
- ONLY use information from the provided context to answer
- If the context doesn't contain the answer, say "I don't have that information in my documents"
- Cite which source file the information came from
- Be concise and direct
- Use simple language suitable for beginners"""},
            {"role": "user", "content": f"""Context from documents:
{context}

---

Question: {query}

Answer based on the context above:"""},
        ],
        max_completion_tokens=1000,
    )

    return response.choices[0].message.content


# ===========================================================================
# PART 5: MAIN
# ===========================================================================

def main():
    print("=" * 55)
    print("  RAG Agent (Step 7)")
    print("  Ask questions about your documents!")
    print("  Type 'quit' to exit")
    print("  Type 'docs' to see loaded documents")
    print("  Type 'chunks' to see all chunks")
    print("=" * 55)
    print()

    # --- Build the knowledge base ---
    print("  Loading and processing documents...")
    documents = load_documents(DOCUMENTS_DIR)
    chunks = chunk_documents(documents)

    vector_store = SimpleVectorStore()
    vector_store.add_chunks(chunks)
    print()

    print("  Try asking:")
    print("    'Who created Python and when?'")
    print("    'What is the difference between Django and Flask?'")
    print("    'Explain the three types of machine learning'")
    print("    'What is the ReAct pattern in agentic AI?'")
    print()

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "docs":
            for doc in documents:
                print(f"  - {doc['filename']} ({len(doc['content'])} chars)")
            print()
            continue
        if user_input.lower() == "chunks":
            for i, chunk in enumerate(chunks):
                preview = chunk["text"][:60] + "..."
                print(f"  [{i}] {chunk['source']}: {preview}")
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
