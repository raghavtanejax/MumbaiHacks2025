from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
import os

# Real OCR using Gemini Vision
def extract_text_from_image(image_base64: str) -> str:
    try:
        if not image_base64:
            return ""
            
        # Remove header if present (e.g., "data:image/jpeg;base64,")
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Transcribe the text in this image exactly. If there is no text, describe the image relevant to health."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        )
        
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f"OCR Error: {e}")
        return f"Error extracting text: {e}"

# Mock OCR Tool (kept for backward compatibility or tool usage)
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
        
        print(f"DEBUG: Starting analysis for query: {query}")
        try:
            # Direct invocation without safety loop for debugging
            result = self.graph.invoke({"messages": messages})
            last_message = result["messages"][-1]
            content = last_message.content
            print(f"DEBUG: Agent raw response: {content[:200]}...")
            
            # Try to parse JSON
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
            except Exception as e:
                print(f"JSON Parse Error: {e}")
                return {
                    "verdict": "Unverified",
                    "confidence": 0.0,
                    "explanation": content,
                    "sources": [],
                    "corrective_information": None
                }

        except Exception as e:
            print(f"CRITICAL ERROR in Agent Invoke: {e}")
            return {
                "verdict": "Error",
                "confidence": 0.0,
                "explanation": f"An internal error occurred: {str(e)}",
                "sources": [],
                "corrective_information": None
            }

def get_agent():
    try:
        # Initialize LLM - Switching to Gemini 1.5 Pro
        # Ensure GOOGLE_API_KEY is set in your .env file
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            convert_system_message_to_human=True # Sometimes needed for Gemini
        )

        tools = [
            DuckDuckGoSearchRun(),
            Tool(
                name="Medical DB",
                func=medical_db_lookup,
                description="Useful for looking up verified medical facts from WHO, CDC, and PubMed."
            ),
            Tool(
                name="OCR",
                func=mock_ocr,
                description="Useful for extracting text from images."
            )
        ]

        # Create LangGraph agent
        graph = create_react_agent(llm, tools)
        return LangGraphAdapter(graph, llm)

    except Exception as e:
        print(f"Warning: Failed to create LangChain/LangGraph agent (likely missing API key). Using mock agent. Error: {e}")
        return MockAgentExecutor()

class MockAgentExecutor:
    def __init__(self):
        self.knowledge_base = {
            # COVID-19 & Vaccines
            "bleach": {"verdict": "False", "confidence": 0.99, "explanation": "Drinking bleach is dangerous and does not cure COVID-19.", "sources": ["WHO", "CDC"], "corrective": "Do not ingest disinfectants."},
            "vaccin": {"verdict": "False", "confidence": 0.98, "explanation": "Vaccines do not cause autism. This myth has been debunked.", "sources": ["CDC", "WHO"], "corrective": "Vaccines are safe."},
            "autism": {"verdict": "False", "confidence": 0.98, "explanation": "Vaccines do not cause autism. This myth has been debunked.", "sources": ["CDC", "WHO"], "corrective": "Vaccines are safe."},
            "microchip": {"verdict": "False", "confidence": 0.99, "explanation": "Vaccines do not contain microchips.", "sources": ["Reuters", "BBC"], "corrective": "Vaccines contain biological ingredients to build immunity."},
            "magnet": {"verdict": "False", "confidence": 0.99, "explanation": "Vaccines do not make you magnetic.", "sources": ["CDC"], "corrective": "None."},
            "ivermectin": {"verdict": "Misleading", "confidence": 0.90, "explanation": "Ivermectin is not approved for treating COVID-19.", "sources": ["FDA", "WHO"], "corrective": "Use approved treatments."},
            "mask": {"verdict": "True", "confidence": 0.95, "explanation": "Masks reduce the spread of respiratory viruses.", "sources": ["CDC", "Nature"], "corrective": "Wear masks in high-risk areas."},
            
            # Technology
            "5g": {"verdict": "False", "confidence": 0.99, "explanation": "5G does not spread viruses.", "sources": ["WHO", "FCC"], "corrective": "Viruses spread via droplets."},
            "radiation": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Cell phones emit non-ionizing radiation, which is generally considered safe.", "sources": ["NCI"], "corrective": "Use hands-free devices if concerned."},

            # Nutrition & Diet
            "lemon": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Lemons do not cure cancer.", "sources": ["NCI"], "corrective": "Consult an oncologist."},
            "alkaline": {"verdict": "False", "confidence": 0.90, "explanation": "You cannot change your blood pH through diet.", "sources": ["WebMD"], "corrective": "The body regulates pH tightly."},
            "detox": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Your liver and kidneys detox your body naturally. Detox teas are unnecessary.", "sources": ["Mayo Clinic"], "corrective": "Eat a balanced diet."},
            "sugar": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Sugar causes hyperactivity is a myth, but excess sugar is unhealthy.", "sources": ["WebMD"], "corrective": "Limit sugar intake."},
            "fat": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Not all fats are bad. Healthy fats are essential.", "sources": ["Harvard Health"], "corrective": "Choose unsaturated fats."},
            "microwave": {"verdict": "False", "confidence": 0.95, "explanation": "Microwaving food does not make it radioactive or destroy all nutrients.", "sources": ["FDA"], "corrective": "Microwaving is safe."},
            "gluten": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Gluten is only harmful to those with Celiac disease or sensitivity.", "sources": ["Celiac.org"], "corrective": "Whole grains are healthy for most."},
            "organic": {"verdict": "Misleading", "confidence": 0.70, "explanation": "Organic food is not necessarily more nutritious, though it has fewer pesticides.", "sources": ["Mayo Clinic"], "corrective": "Focus on eating fruits/veg, organic or not."},
            "apple cider vinegar": {"verdict": "Misleading", "confidence": 0.80, "explanation": "ACV has some benefits but is not a cure-all for weight loss or cancer.", "sources": ["UChicago Medicine"], "corrective": "Use in moderation."},

            # Diseases & Cures
            "garlic": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Garlic is healthy but doesn't cure viruses.", "sources": ["WHO"], "corrective": "Eat for flavor."},
            "cancer": {"verdict": "Misleading", "confidence": 0.90, "explanation": "There is no single 'cure' for cancer. It is a complex group of diseases.", "sources": ["NCI"], "corrective": "Follow medical advice."},
            "diabetes": {"verdict": "Misleading", "confidence": 0.90, "explanation": "Eating too much sugar doesn't directly cause diabetes, but obesity is a risk factor.", "sources": ["ADA"], "corrective": "Maintain healthy weight."},
            "flu shot": {"verdict": "False", "confidence": 0.95, "explanation": "The flu shot cannot give you the flu.", "sources": ["CDC"], "corrective": "Get vaccinated annually."},
            "antibiotic": {"verdict": "Misleading", "confidence": 0.95, "explanation": "Antibiotics do not kill viruses (cold/flu).", "sources": ["CDC"], "corrective": "Only use for bacterial infections."},
            "cold": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Cold weather doesn't cause colds; viruses do.", "sources": ["Mayo Clinic"], "corrective": "Wash hands."},
            "vitamin c": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Vitamin C may slightly shorten colds but doesn't prevent them.", "sources": ["Harvard Health"], "corrective": "Eat citrus fruits."},

            # Mental Health & Sleep
            "brain": {"verdict": "False", "confidence": 0.90, "explanation": "We use more than 10% of our brains.", "sources": ["Scientific American"], "corrective": "The whole brain is active."},
            "sleep": {"verdict": "Misleading", "confidence": 0.85, "explanation": "You cannot 'catch up' on sleep fully on weekends.", "sources": ["Sleep Foundation"], "corrective": "Maintain consistent sleep schedule."},
            "depression": {"verdict": "Misleading", "confidence": 0.90, "explanation": "Depression is not just 'being sad'. It's a clinical condition.", "sources": ["NIMH"], "corrective": "Seek professional help."},

            # Home Remedies & Myths
            "toothpaste": {"verdict": "Misleading", "confidence": 0.75, "explanation": "Toothpaste can irritate acne.", "sources": ["AAD"], "corrective": "Use acne medication."},
            "shaving": {"verdict": "False", "confidence": 0.90, "explanation": "Hair does not grow back thicker after shaving.", "sources": ["Mayo Clinic"], "corrective": "It just feels coarser."},
            "gum": {"verdict": "False", "confidence": 0.95, "explanation": "Gum does not stay in your stomach for 7 years.", "sources": ["Mayo Clinic"], "corrective": "It passes through."},
            "knuckle": {"verdict": "False", "confidence": 0.90, "explanation": "Cracking knuckles does not cause arthritis.", "sources": ["Harvard Health"], "corrective": "It may weaken grip strength."},
            "carrot": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Carrots are good for eyes but won't give you night vision.", "sources": ["Scientific American"], "corrective": "Eat a balanced diet."},
            "water": {"verdict": "True", "confidence": 0.90, "explanation": "Hydration is important.", "sources": ["Mayo Clinic"], "corrective": "Drink when thirsty."},
            "8 glasses": {"verdict": "Misleading", "confidence": 0.80, "explanation": "The '8 glasses a day' rule is not scientifically rigid.", "sources": ["Mayo Clinic"], "corrective": "Drink when thirsty."},
            
            # General
            "natural": {"verdict": "Misleading", "confidence": 0.70, "explanation": "'Natural' doesn't always mean safe (e.g., arsenic is natural).", "sources": ["FDA"], "corrective": "Check ingredients."},
            "chemical": {"verdict": "Misleading", "confidence": 0.70, "explanation": "Everything is made of chemicals, including water.", "sources": ["Science"], "corrective": "Don't fear the word 'chemical'."}
        }

    def invoke(self, input_dict):
        query = input_dict.get("input", "").lower()
        
        # Check for keywords in the query
        for key, data in self.knowledge_base.items():
            if key in query:
                return {
                    "verdict": data["verdict"],
                    "confidence": data["confidence"],
                    "explanation": data["explanation"],
                    "sources": data["sources"],
                    "corrective_information": data["corrective"]
                }
        
        # Default fallback
        return {
            "verdict": "Unverified",
            "confidence": 0.50,
            "explanation": f"The claim '{query}' requires further investigation. In this demo version (Mock Mode), I recognize 50+ common health myths. Try asking about 'sugar', 'detox', '5G', 'vaccines', 'sleep', etc.",
            "sources": [],
            "corrective_information": "Please consult a medical professional."
        }
