from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
import os

# Mock OCR Tool
def mock_ocr(image_path: str) -> str:
    return "Extracted text from image: Drinking bleach cures COVID-19."

# Mock Medical DB Tool
def medical_db_lookup(query: str) -> str:
    if "bleach" in query.lower():
        return "WHO: Do NOT drink bleach. It is dangerous and does not cure COVID-19."
    return "No specific medical record found."


class LangGraphAdapter:
    def __init__(self, graph, llm):
        self.graph = graph
        self.llm = llm

    def validate_response(self, response_content: str) -> tuple[bool, str]:
        """
        Uses the LLM to validate the response.
        Returns (is_valid, feedback).
        """
        validation_prompt = f"""
        You are a Quality Assurance Safety Agent for a health misinformation detector.
        Review the following analysis to ensure it meets these criteria:
        1. Safety: Does not recommend dangerous actions.
        2. Grounding: Cites sources (e.g., WHO, CDC, Medical DB) if making factual claims.
        3. Tone: Professional and neutral.
        4. Structure: Returns a JSON object with 'verdict', 'confidence', 'explanation', 'sources', 'corrective_information'.

        Analysis to review:
        {response_content}

        If it passes, return exactly: PASS
        If it fails, return: FAIL: <brief explanation of what to fix>
        """
        try:
            # We use the same LLM for validation for simplicity
            validation_response = self.llm.invoke(validation_prompt).content
            if validation_response.strip().startswith("PASS"):
                return True, ""
            else:
                return False, validation_response
        except Exception as e:
            print(f"Validation error: {e}")
            return True, "" # Fail open if validation errors out

    def invoke(self, input_dict):
        query = input_dict.get("input", "")
        messages = [HumanMessage(content=query)]
        
        max_retries = 3
        current_try = 0
        
        while current_try < max_retries:
            result = self.graph.invoke({"messages": messages})
            last_message = result["messages"][-1]
            content = last_message.content
            
            # Validate
            is_valid, feedback = self.validate_response(content)
            
            if is_valid:
                # Try to parse JSON from the content
                import json
                import re
                try:
                    match = re.search(r'\{.*\}', content, re.DOTALL)
                    if match:
                        return json.loads(match.group(0))
                    else:
                        return {
                            "verdict": "Unverified",
                            "confidence": 0.0,
                            "explanation": content,
                            "sources": [],
                            "corrective_information": None
                        }
                except:
                     return {
                        "verdict": "Unverified",
                        "confidence": 0.0,
                        "explanation": content,
                        "sources": [],
                        "corrective_information": None
                    }
            
            # If invalid, add feedback and retry
            print(f"Safety Check Failed (Attempt {current_try+1}/{max_retries}): {feedback}")
            messages.append(last_message) # Add the agent's bad response
            messages.append(HumanMessage(content=f"Safety Agent Feedback: {feedback}. Please regenerate the response fixing these issues."))
            current_try += 1
            
        # If we run out of retries, return the last one (or a failure message)
        return {
            "verdict": "Unverified",
            "confidence": 0.0,
            "explanation": "Safety Agent rejected the response after multiple attempts. Please try again.",
            "sources": [],
            "corrective_information": None
        }

def get_agent():
    try:
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        tools = [
            DuckDuckGoSearchRun(),
            Tool(
                name="Medical DB",
                func=medical_db_lookup,
                description="Useful for looking up verified medical facts."
            ),
            Tool(
                name="OCR",
                func=mock_ocr,
                description="Useful for extracting text from images."
            )
        ]

        template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: A JSON object with the following keys:
- "verdict": "True", "False", "Misleading", or "Unverified"
- "confidence": a float between 0.0 and 1.0
- "explanation": a detailed explanation of the analysis
- "sources": a list of source names or URLs
- "corrective_information": specific facts to counter misinformation (if applicable, else null)

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

        # Create LangGraph agent
        graph = create_react_agent(llm, tools)
        return LangGraphAdapter(graph, llm)

    except Exception as e:
        print(f"Warning: Failed to create LangChain/LangGraph agent (likely missing API key). Using mock agent. Error: {e}")
        return MockAgentExecutor()

class MockAgentExecutor:
    def invoke(self, input_dict):
        query = input_dict.get("input", "").lower()
        
        if "bleach" in query:
            return {
                "verdict": "False",
                "confidence": 0.99,
                "explanation": "Drinking bleach is extremely dangerous and does NOT cure COVID-19. Health authorities warn against ingesting disinfectants.",
                "sources": ["WHO", "CDC"],
                "corrective_information": "Do not drink bleach. Seek medical attention if ingested."
            }
        elif "lemon" in query and "cancer" in query:
            return {
                "verdict": "Misleading",
                "confidence": 0.85,
                "explanation": "While lemons are healthy, there is no scientific evidence that they cure cancer. This is a common myth.",
                "sources": ["National Cancer Institute", "Snopes"],
                "corrective_information": "Cancer treatment requires medical intervention. Consult an oncologist."
            }
        elif "vaccin" in query or "autism" in query:
            return {
                "verdict": "False",
                "confidence": 0.98,
                "explanation": "Extensive research has shown no link between vaccines and autism. The original study suggesting this was retracted and debunked.",
                "sources": ["CDC", "Autism Speaks", "WHO"],
                "corrective_information": "Vaccines are safe and effective at preventing disease."
            }
        elif "5g" in query:
            return {
                "verdict": "False",
                "confidence": 0.95,
                "explanation": "5G radio waves cannot spread viruses. COVID-19 is caused by a virus, not radio waves.",
                "sources": ["WHO", "FCC"],
                "corrective_information": "Viruses spread through respiratory droplets, not cellular networks."
            }
        elif "garlic" in query:
            return {
                "verdict": "Misleading",
                "confidence": 0.80,
                "explanation": "Garlic has some health benefits, but it is not a miracle cure for viruses or major diseases.",
                "sources": ["NCCIH", "WHO"],
                "corrective_information": "Eat garlic for flavor, not as a substitute for medicine."
            }
        elif "water" in query:
             return {
                "verdict": "True",
                "confidence": 0.90,
                "explanation": "Staying hydrated is generally good for health, though specific miracle claims should be viewed with skepticism.",
                "sources": ["Mayo Clinic"],
                "corrective_information": None
            }
        else:
            return {
                "verdict": "Unverified",
                "confidence": 0.50,
                "explanation": f"The claim '{query}' requires further investigation. In this demo version (Mock Mode), I only recognize specific topics like 'bleach', 'vaccines', '5G', 'garlic', and 'lemons'.",
                "sources": [],
                "corrective_information": "Please consult a medical professional."
            }
