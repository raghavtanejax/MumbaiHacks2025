import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

def test_gemini_api():
    api_key = os.getenv("GOOGLE_API_KEY")
    print(f"DEBUG: Checking API Key...")
    
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment variables.")
        print("Please ensure you have a .env file in the backend directory with GOOGLE_API_KEY=your_key_here")
        return

    # Mask the key for security in logs, show first/last 4 chars
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
    print(f"DEBUG: Found API Key: {masked_key}")

    try:
        print("DEBUG: Attempting to connect to Google Gemini API...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            google_api_key=api_key
        )
        
        message = HumanMessage(content="Hello, are you working? Reply with 'Yes, I am working!'")
        response = llm.invoke([message])
        
        print("\nSUCCESS! The API is working correctly.")
        print(f"Response from Gemini: {response.content}")
        
    except Exception as e:
        print("\nFAILURE: The API connection failed.")
        print(f"Error details: {e}")
        print("\nCommon causes:")
        print("1. Invalid API Key.")
        print("2. API Key does not have 'Generative Language API' enabled in Google Cloud Console.")
        print("3. Billing is not enabled (though Gemini 1.5 Pro has a free tier, it might require setup).")
        print("4. Network issues blocking the request.")

if __name__ == "__main__":
    test_gemini_api()
