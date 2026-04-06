import os
from typing import TypedDict
from google import genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 1. Initialize your Gemini Client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# 2. Define the State
class BrigadeState(TypedDict):
    user_input: str
    target_domain: str
    current_draft: str
    qa_feedback: list[str]
    final_output: str

# 3. Build the Node
def node_the_butcher(state: BrigadeState):
    print("--- THE BUTCHER IS CHOPPING ---")
    
    problem = state["user_input"]
    
    system_instruction = (
        "You are The Butcher. You strip problems of all assumptions and analogies "
        "to reveal the fundamental truths, then rebuild a solution from scratch."
    )
    
    response = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=f"{system_instruction}\n\nDeconstruct this: {problem}",
    )
    
    return {"current_draft": response.text}