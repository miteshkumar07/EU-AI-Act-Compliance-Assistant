import os
import json
from google import genai
from google.genai import types
from google.genai import errors

# Import retrieval logic safely
try:
    from retrieval_service import retrieve_top_k 
except ImportError:
    raise ImportError("Ensure retrieval_service.py is present in the same directory.")

# Initialize the Gemini Client
client = genai.Client()

def generate_rag_answer(user_query: str, collection_name: str = "document_aware_by_article", k: int = 2):
    # Log operational inputs
    print(f"\n" + "="*50)
    print(f"[INPUT QUERY]: '{user_query}'")
    print(f"[TARGET COLLECTION]: {collection_name} | Top K: {k}")
    print("="*50)
    
    # 1. Defensive Retrieval Execution
    try:
        retrieved_chunks = retrieve_top_k(query=user_query, collection_name=collection_name, k=k)
    except Exception as e:
        print(f"[CRITICAL ERROR] ChromaDB retrieval layer collapsed: {e}")
        return "System Error: Unable to extract data from local knowledge base."

    # 2. Handle Edge Case: Zero Documents Retrieved
    if not retrieved_chunks:
        print("[WARN] ChromaDB returned 0 matching results for this vector.")
        context_text = "No relevant text chunks found in the database."
    else:
        # Build pristine string context block
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks):
            block = (
                f"--- [Match {i+1}] Source: {chunk.get('source', 'Unknown')} "
                f"| Section: {chunk.get('section', 'N/A')} ---\n"
                f"{chunk.get('text', '').strip()}\n"
            )
            context_blocks.append(block)
        context_text = "\n".join(context_blocks)

    # Logging: Exactly what text gets stuffed into the Prompt Envelope
    print(f"\n[LOGGING RETRIEVED CONTEXT SENT TO LLM]:")
    print("-" * 50)
    print(context_text if retrieved_chunks else "[Empty Context Window Passed]")
    print("-" * 50)

    # 3. Create the System Instructions & Prompts
    system_instruction = (
        "You are an expert European AI legal advisor.\n"
        "Answer the user question strictly and only using the provided legal context.\n"
        "If the answer cannot be confidently deduced from the text, explicitly state: "
        "\"I do not have enough legal context to answer this question.\"\n"
        "Do not invent external rules. Always cite specific Articles or Sections used."
    )
    
    user_prompt = f"Retrieved Legal Context:\n{context_text}\n\nQuestion: {user_query}"

    # 4. Protected LLM Generation Layer
    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1 # Lock down randomness
            )
        )
        
        # Log and return output
        print(f"\n[LOGGING RAW LLM RESPONSE GENERATED]:")
        print("." * 50)
        print(response.text)
        print("." * 50)
        return response.text

    except errors.ServerError as e:
        # Catch 503 high demand, 500 internals, etc.
        error_msg = f"[API FAILURE] Google Server side exception (HTTP {e.code}): {e.message}"
        print(error_msg)
        return "Generation Error: The remote LLM infrastructure is congested. Please re-run."
        
    except errors.ClientError as e:
        # Catch 400 bad args, 403 bad keys, 429 quota exhaustion
        error_msg = f"[API FAILURE] Client request invalidation (HTTP {e.code}): {e.message}"
        print(error_msg)
        return f"Configuration Error: Request rejected by API provider ({e.code})."
        
    except Exception as e:
        print(f"[UNKNOWN ERROR]: Execution interrupted: {e}")
        return "An unexpected pipeline failure occurred."

import time
if __name__ == "__main__":
    # Smoke-test execution
    test_question = ["What are the specific criteria to qualify an AI system as high-risk under Article 6?", "If a provider complies with the codes of practice mentioned in Article 98, does that automatically satisfy their obligations under Article 53?", "Who is considered a 'deployer' under the Act, and what are their obligations regarding human oversight?", "Does the copyright summary obligation apply to all general-purpose AI models, or only those with systemic risks?", "What are the penalties for violating the prohibitions on AI practices listed in Article 5?", "What does the Act say about the use of AI systems for real-time biometric identification in publicly accessible spaces for the purpose of searching for kidnapping victims?", "Are simple legacy software systems or basic Excel macros covered under the definition of an Artificial Intelligence System?", "What specific exemptions exist for AI systems developed or used exclusively for military, defense, or national security purposes?", "What are the reporting timelines if a high-risk AI system causes a serious incident?", "What are the compliance fines for a small business using an AI chatbot in California?", "How does the EU AI Act regulate the carbon footprint and environmental impact of training large language models?", "What is the exact step-by-step registration process for submitting a system to the EU database website interface?"]
    for i in range(len(test_question)):
        print(f"\n\n[TEST CASE {i+1}]")
        print("-"*50)
        print(f"Query: {test_question[i]}")
        print("-"*50)
        answer = generate_rag_answer(test_question[i])
        print(f"\n[FINAL ANSWER GENERATED]:\n{answer}")
        print("="*50 + "\n\n")
        time.sleep(60)  # Add a small delay between test cases