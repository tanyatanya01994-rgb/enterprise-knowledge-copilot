import os
from dotenv import load_dotenv
from google import genai


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set in the .env file."
    )


# ---------------------------------------------------------
# Create Gemini client
# ---------------------------------------------------------

client = genai.Client(api_key=API_KEY)


# ---------------------------------------------------------
# Available models
# ---------------------------------------------------------

MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash"
]


# ---------------------------------------------------------
# Query Rewriting Function
# ---------------------------------------------------------

def rewrite_query(query, conversation_history=None):

    if not conversation_history:
        return query.strip()

    history_text = "\n".join(conversation_history)

    prompt = f"""
You are a query rewriting component for an
enterprise document intelligence system.

Your job is to convert the user's latest question
into a standalone search query.

Rules:

1. Preserve the exact meaning of the user's question.

2. Use previous conversation ONLY to resolve references
   such as:
   - "it"
   - "they"
   - "that policy"
   - "what about interns"
   - "what about them"

3. Do NOT answer the question.

4. Do NOT add information that is not supported
   by the conversation.

5. Return ONLY the rewritten search query.

Conversation:
{history_text}

Latest user question:
{query}

Standalone search query:
"""

    # Try the models one by one
    for model_name in MODELS:

        try:
            print(f"\nTrying model: {model_name}")

            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            if response.text:
                print(f"Model used: {model_name}")
                return response.text.strip()

        except Exception as error:

            print(f"Model unavailable: {model_name}")
            print("Trying the next model...")

    # If all models fail
    raise RuntimeError(
        "All Gemini models are currently unavailable. "
        "Please try again later."
    )


# ---------------------------------------------------------
# Test Program
# ---------------------------------------------------------

def main():

    history = [
        "User: How many annual leave days do employees get?",
        "Assistant: Eligible full-time employees receive "
        "18 days of annual leave per calendar year."
    ]

    query = input("Enter your follow-up question: ")

    rewritten = rewrite_query(
        query,
        history
    )

    print("\n" + "=" * 60)
    print("QUERY REWRITING")
    print("=" * 60)

    print("\nOriginal question:")
    print(query)

    print("\nRewritten search query:")
    print(rewritten)

    print("\n" + "=" * 60)


# ---------------------------------------------------------
# Program Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()