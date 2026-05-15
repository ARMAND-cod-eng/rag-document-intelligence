from together import Together
from app.core.config import settings

# Initialize Together AI client
client = Together(api_key=settings.TOGETHER_API_KEY)

def build_prompt(question: str, chunks: list) -> str:
    """
    Build the prompt sent to LLaMA.
    
    This is the most important part of RAG quality.
    We give the model ONLY the retrieved chunks as context
    so it cannot hallucinate — it must answer from your documents.
    """
    context = "\n\n".join([
        f"[Source: {chunk['source']} | Page: {chunk['page']} | "
        f"Relevance: {chunk['relevance_score']}]\n{chunk['content']}"
        for chunk in chunks
    ])

    prompt = f"""You are a precise document assistant. 
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I could not find this information in the provided documents."
Always cite which source and page your answer comes from.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    return prompt

def generate_answer(question: str, chunks: list) -> dict:
    """
    Send question + retrieved chunks to LLaMA → get cited answer.
    """
    if not chunks:
        return {
            "answer": "No relevant content found in the documents for your question.",
            "sources": [],
            "model": settings.LLM_MODEL
        }

    prompt = build_prompt(question, chunks)

    # Call LLaMA via Together AI
    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS
    )

    answer = response.choices[0].message.content.strip()

    # Extract unique sources cited
    sources = list(set([
        f"{chunk['source']} (page {chunk['page']})"
        for chunk in chunks
    ]))

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks),
        "model": settings.LLM_MODEL
    }