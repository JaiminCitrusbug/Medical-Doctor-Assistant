import os
from openai import OpenAI
from retriever import retrieve_similar_chunks

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def determine_retrieval_query(user_query, history):
    """
    Use LLM to dynamically determine the best query for retrieval based on conversation context.
    The LLM analyzes the conversation and determines what should be searched in the vector database.
    """
    # Convert history to chat format
    chat_history = [{"role": m["role"], "content": m["content"]} for m in history]
    
    # Create a prompt for the LLM to determine the retrieval query
    query_determination_prompt = {
        "role": "system",
        "content": """You are a query analyzer for a medical information retrieval system. 
Your task is to analyze the conversation and determine the BEST search query to use for retrieving relevant information from the knowledge base.

CRITICAL RULES:
1. If the user is confirming a correction (e.g., responding "yes" to "Did you mean CIPROTAB?"), extract the CORRECTED TERM (e.g., "CIPROTAB") from the conversation history.
2. If the user is asking a new question, analyze it for potential misspellings and try to normalize it:
   - If you recognize a misspelling (e.g., "antrovast" → likely "atorvastatin"), use the corrected version
   - If you recognize a partial match (e.g., "antrovast" contains "atorvast"), expand to full term
   - Consider phonetic similarities and common drug/product name patterns
3. If the user is referring to something mentioned earlier (using pronouns like "it", "that", "this"), extract the actual entity name from the conversation history.
4. Always return ONLY the search query term(s) - no explanations, no questions, just the query string.
5. Focus on extracting the specific product name, drug name, or medical term that should be searched.

Examples:
- User: "I need details on Ciprteb" → Return: "CIPROTAB" (corrected spelling)
- User: "antrovast" → Return: "atorvastatin" or "ATORITIC" (normalized/corrected)
- User: "antrovastic" → Return: "atorvastatin" (corrected)
- User: "Tell me about antibiotics" → Return: "antibiotics"
- User: "What about it?" (after discussing CIPROTAB) → Return: "CIPROTAB"
- User: "I need details on Ciprotab" → Return: "CIPROTAB"

IMPORTANT: Be proactive in correcting obvious misspellings. If you recognize a drug/product name even with typos, use the corrected version.

Return ONLY the search query, nothing else."""
    }
    
    # Build messages for query determination
    messages = [query_determination_prompt] + chat_history + [
        {"role": "user", "content": f"Given the conversation above, what should be the search query for the current user message: '{user_query}'?\n\nReturn ONLY the search query:"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=50,  # Short response - just the query
        )
        
        retrieval_query = response.choices[0].message.content.strip()
        
        # Clean up the response - remove quotes if present
        retrieval_query = retrieval_query.strip('"').strip("'").strip()
        
        # If LLM returns something like "The search query should be: X", extract X
        if ":" in retrieval_query and len(retrieval_query.split(":")) > 1:
            retrieval_query = retrieval_query.split(":")[-1].strip()
        
        # Fallback: if LLM returns something unexpected, use original query
        if not retrieval_query or len(retrieval_query) < 2:
            retrieval_query = user_query
        
        print(f"🤖 LLM determined retrieval query: '{retrieval_query}' (from user: '{user_query}')")
        return retrieval_query
        
    except Exception as e:
        print(f"⚠️  Error in LLM query determination: {e}. Using original query.")
        return user_query

def generate_answer(user_query, history):
    """Generate context-aware answer using chat history and RAG."""
    # Use LLM to dynamically determine the best query for retrieval
    retrieval_query = determine_retrieval_query(user_query, history)
    
    # Retrieve more chunks to have better candidates for fuzzy matching
    # This helps when user misspells - we get more potential matches to analyze
    retrieved_chunks = retrieve_similar_chunks(retrieval_query, top_k=5)
    context_text = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])

    # Convert Streamlit history to OpenAI chat format
    chat_history = [{"role": m["role"], "content": m["content"]} for m in history]

    system_prompt = {
    "role": "system",
    "content": (
        "You are a reliable and context-aware medical assistant that supports doctors in emergencies. "
        "Use the entire conversation history and the provided context to give accurate, concise, and safe answers. "
        "Never guess or invent information—if uncertain, say so clearly.\n\n"

        "CRITICAL: Enhanced Fuzzy Matching and Correction Behavior:\n"
        "- ALWAYS analyze the user's query for potential misspellings, typos, or phonetic similarities.\n"
        "- When the user asks about something that doesn't exactly match the context, IMMEDIATELY search for the closest match.\n"
        "- Consider phonetic similarity (e.g., 'antrovast' sounds like 'atorvastatin', 'antrovastic' → 'atorvastatin').\n"
        "- Consider partial matches (e.g., 'antrovast' contains 'atorvast' which is close to 'atorvastatin').\n"
        "- Consider common misspellings: letter swaps, dropped letters, added letters, similar-sounding words.\n"
        "- Look for product names (e.g., 'ATORITIC') AND active ingredients (e.g., 'Atorvastatin') in the context.\n"
        "- Even if similarity scores are low, if you can identify a reasonable match, suggest it.\n\n"

        "Correction Message Format (MANDATORY):\n"
        "- When suggesting a correction, ALWAYS use this format: \"I don't have information about '[user_query]'. Did you mean '[suggested_match]'?\"\n"
        "- Example: User says \"antrovast\" → Respond: \"I don't have information about 'antrovast'. Did you mean 'ATORITIC' or 'Atorvastatin'?\"\n"
        "- Example: User says \"kiprteb\" → Respond: \"I don't have information about 'kiprteb'. Did you mean 'CIPROTAB'?\"\n"
        "- Always include the user's original query in quotes in your response.\n\n"

        "Matching Strategy:\n"
        "1. First, check if the user query exactly matches any product name, brand name, or active ingredient in the context.\n"
        "2. If no exact match, look for phonetic similarities (how it sounds).\n"
        "3. If no phonetic match, look for partial string matches (substrings).\n"
        "4. If no partial match, look for similar therapeutic classes or related terms.\n"
        "5. If you find ANY reasonable match (even with low confidence), suggest it using the format above.\n"
        "6. Only say 'I couldn't find that' if there's truly NO reasonable match in the context.\n\n"

        "On user confirmation (yes, yeah, correct, exactly, right, that's it, etc.):\n"
        "- Immediately proceed using the last suggested match from the assistant's clarification. "
        "- Provide complete information about that entity from the context. "
        "- Never respond with 'I couldn't find that' after a confirmed match unless the term truly does not exist in context.\n\n"

        "If the user follows up with a related or generalized term (e.g., 'antibiotic', 'cardiac drugs', 'painkillers'), "
        "interpret it contextually. Dynamically identify related entities within the same therapeutic class or purpose. "
        "Provide meaningful information or examples from the data that align with that category.\n\n"

        "Dynamic context memory rules:\n"
        "- Remember the last clarification and the confirmed term in the conversation. "
        "- Use that stored term when the user later refers with pronouns ('it', 'that', 'this') or follow-ups ('What about it?'). "
        "- Keep the flow natural and consistent—never lose or reset confirmed context within the same session.\n\n"

        "Matching and reasoning priorities:\n"
        "1. Exact match → respond directly with information.\n"
        "2. Fuzzy / misspelled match → suggest correction using the format: \"I don't have information about '[query]'. Did you mean '[match]'?\"\n"
        "3. Related term match → provide connected examples or explain related items.\n"
        "4. No match → politely say: \"I couldn't find that in my current data. Could you please rephrase or clarify what you meant?\"\n\n"

        "Maintain a professional, empathetic tone suitable for medical professionals. "
        "Be concise, factual, and medically safe in all responses.\n\n"

        f"Context from knowledge base:\n{context_text}\n\n"
        "Remember: Be PROACTIVE in suggesting corrections. Even if the match isn't perfect, if it's reasonable, suggest it!"
    ),
}




    messages = [system_prompt] + chat_history + [{"role": "user", "content": user_query}]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()

