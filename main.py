import os
import sys
from dotenv import load_dotenv
from google import genai

load_dotenv()

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    model_name = "gemini-2.0-flash-001"
    prompt = sys.argv[1]
    verbose = sys.argv.pop() == "--verbose"
    messages = [
        genai.types.Content(
            role="user",
            parts=[genai.types.Part(text=prompt)]
        )
    ]
    
    response = client.models.generate_content(
        model=model_name,
        contents=messages
    )
    
    print("Response text:")
    print(response.text)

    if verbose:
        print(f"\nUser prompt: {prompt}")
        print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
        print(f"Response tokens: {response.usage_metadata.candidates_token_count}")


if __name__ == "__main__":
    main()
