from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
import os
from dotenv import load_dotenv

load_dotenv()

import time

# Real OCR using Gemini Vision
def extract_text_from_image(image_base64: str) -> str:
    try:
        start_time = time.time()
        print("DEBUG: Starting OCR extraction...")
        
        if not image_base64:
            return ""
            
        # Remove header if present (e.g., "data:image/jpeg;base64,")
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "Error: GOOGLE_API_KEY not found in environment."

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro", 
            temperature=0,
            google_api_key=api_key
        )
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Transcribe the text in this image exactly. If there is no text, describe the image relevant to health."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        )
        
        response = llm.invoke([message])
        end_time = time.time()
        print(f"DEBUG: OCR completed in {end_time - start_time:.2f} seconds")
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
        api_key = os.getenv("GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            google_api_key=api_key,
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
            # --- NUTRITION & DIET ---
            "sugar": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Sugar itself doesn't cause hyperactivity, but excess intake leads to obesity and diabetes risk.", "sources": ["CDC", "WebMD"], "corrective": "Limit added sugars."},
            "brown sugar": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Brown sugar is not significantly healthier than white sugar; it just contains molasses.", "sources": ["Mayo Clinic"], "corrective": "Treat both as added sugar."},
            "honey": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Honey is natural but is still sugar. It has minor antimicrobial properties but affects blood sugar.", "sources": ["Healthline"], "corrective": "Use in moderation."},
            "agave": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Agave nectar is very high in fructose, which can strain the liver.", "sources": ["Healthline"], "corrective": "It is not a 'healthy' superfood sweetener."},
            "aspartame": {"verdict": "False", "confidence": 0.90, "explanation": "Aspartame is safe for most people and does not cause cancer at normal consumption levels.", "sources": ["FDA", "Cancer.org"], "corrective": "Safe in moderation."},
            "msg": {"verdict": "False", "confidence": 0.90, "explanation": "MSG does not cause headaches in most people ('Chinese Restaurant Syndrome' is largely a myth).", "sources": ["FDA"], "corrective": "Safe flavor enhancer."},
            "gluten": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Gluten is only harmful to those with Celiac disease or sensitivity.", "sources": ["Celiac.org"], "corrective": "Whole grains are healthy for most."},
            "fat": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Not all fats are bad. Unsaturated fats (avocado, nuts) are essential.", "sources": ["Harvard Health"], "corrective": "Avoid trans fats, eat healthy fats."},
            "eggs": {"verdict": "False", "confidence": 0.85, "explanation": "Eggs do not immediately cause heart disease. They are a good source of protein.", "sources": ["Mayo Clinic"], "corrective": "One egg a day is generally safe."},
            "cholesterol": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Dietary cholesterol (like in eggs) has less impact on blood cholesterol than saturated/trans fats.", "sources": ["Heart.org"], "corrective": "Focus on reducing saturated fats."},
            "carbs": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Carbohydrates are the body's main energy source. Complex carbs are healthy.", "sources": ["Mayo Clinic"], "corrective": "Avoid refined carbs, eat whole grains."},
            "protein": {"verdict": "Misleading", "confidence": 0.80, "explanation": "More protein isn't always better. Excess protein is stored as fat.", "sources": ["WebMD"], "corrective": "Balance protein intake."},
            "organic": {"verdict": "Misleading", "confidence": 0.70, "explanation": "Organic food is not necessarily more nutritious, though it has fewer pesticides.", "sources": ["Mayo Clinic"], "corrective": "Focus on eating fruits/veg, organic or not."},
            "gmo": {"verdict": "False", "confidence": 0.90, "explanation": "GMO foods are currently considered safe to eat by major science organizations.", "sources": ["FDA", "WHO"], "corrective": "GMOs allow for better crop yields."},
            "fresh vs frozen": {"verdict": "False", "confidence": 0.85, "explanation": "Frozen vegetables are often just as nutritious as fresh ones.", "sources": ["Healthline"], "corrective": "Eat whichever helps you eat more veggies."},
            "microwave": {"verdict": "False", "confidence": 0.95, "explanation": "Microwaving food does not destroy all nutrients or make food radioactive.", "sources": ["FDA"], "corrective": "It preserves nutrients well due to short cooking time."},
            "raw food": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Cooking makes some nutrients more absorbable (like lycopene in tomatoes).", "sources": ["Scientific American"], "corrective": "A mix of raw and cooked is best."},
            "detox": {"verdict": "Misleading", "confidence": 0.90, "explanation": "Your liver and kidneys detox your body naturally. Detox teas/diets are unnecessary.", "sources": ["Mayo Clinic"], "corrective": "Eat a balanced diet."},
            "alkaline": {"verdict": "False", "confidence": 0.90, "explanation": "You cannot change your blood pH through diet. The body regulates it tightly.", "sources": ["WebMD"], "corrective": "Eat veggies for nutrients, not pH."},
            "apple cider vinegar": {"verdict": "Misleading", "confidence": 0.80, "explanation": "ACV has some benefits but is not a cure-all for weight loss or cancer.", "sources": ["UChicago Medicine"], "corrective": "Use in moderation."},
            "lemon water": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Lemon water aids hydration but does not 'melt fat' or cure diseases.", "sources": ["Healthline"], "corrective": "Good for hydration."},
            "celery juice": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Celery juice is healthy but not a miracle cure for autoimmune diseases.", "sources": ["Healthline"], "corrective": "Eat whole celery for fiber."},
            "coconut oil": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Coconut oil is high in saturated fat. It's not a heart-health superfood.", "sources": ["Heart.org"], "corrective": "Use sparingly."},
            "sea salt": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Sea salt has the same sodium content as table salt.", "sources": ["Mayo Clinic"], "corrective": "Limit total sodium intake."},
            "himalayan salt": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Pink salt provides trace minerals but not enough to impact health significantly.", "sources": ["WebMD"], "corrective": "It's still salt."},
            "multivitamin": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Most healthy people don't need a multivitamin.", "sources": ["Johns Hopkins"], "corrective": "Get nutrients from food."},
            "vitamin c": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Vitamin C may slightly shorten colds but doesn't prevent them.", "sources": ["Harvard Health"], "corrective": "Eat citrus fruits."},
            "vitamin d": {"verdict": "True", "confidence": 0.90, "explanation": "Many people are deficient in Vitamin D, especially in winter.", "sources": ["CDC"], "corrective": "Check levels with a doctor."},
            "calcium": {"verdict": "True", "confidence": 0.95, "explanation": "Calcium is essential for bone health.", "sources": ["NIH"], "corrective": "Dairy and leafy greens are good sources."},
            "iron": {"verdict": "True", "confidence": 0.95, "explanation": "Iron is needed to prevent anemia.", "sources": ["NIH"], "corrective": "Red meat, beans, spinach."},
            "probiotics": {"verdict": "True", "confidence": 0.85, "explanation": "Probiotics can support gut health.", "sources": ["Cleveland Clinic"], "corrective": "Yogurt and fermented foods."},
            "prebiotics": {"verdict": "True", "confidence": 0.85, "explanation": "Prebiotics feed good gut bacteria.", "sources": ["Mayo Clinic"], "corrective": "Garlic, onions, bananas."},
            "fiber": {"verdict": "True", "confidence": 0.95, "explanation": "Fiber is crucial for digestion and heart health.", "sources": ["Mayo Clinic"], "corrective": "Eat whole grains and veggies."},
            "water": {"verdict": "True", "confidence": 0.90, "explanation": "Hydration is important for all body functions.", "sources": ["Mayo Clinic"], "corrective": "Drink when thirsty."},
            "8 glasses": {"verdict": "Misleading", "confidence": 0.80, "explanation": "The '8 glasses a day' rule is not scientifically rigid.", "sources": ["Mayo Clinic"], "corrective": "Drink when thirsty."},
            "coffee": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Moderate coffee consumption is generally safe and may have antioxidants.", "sources": ["Harvard Health"], "corrective": "Watch out for sugar/cream."},
            "caffeine": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Caffeine dehydrating effect is mild. It counts towards fluid intake.", "sources": ["Mayo Clinic"], "corrective": "Limit to 400mg/day."},
            "energy drinks": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Energy drinks often contain excessive sugar and caffeine.", "sources": ["CDC"], "corrective": "Coffee or tea are better choices."},
            "alcohol": {"verdict": "Misleading", "confidence": 0.90, "explanation": "No amount of alcohol is strictly 'healthy', though moderation is key.", "sources": ["WHO"], "corrective": "Limit intake."},
            "red wine": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Resveratrol in red wine is beneficial, but you can get it from grapes.", "sources": ["Heart.org"], "corrective": "Don't start drinking for health."},
            "chocolate": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Dark chocolate has antioxidants, but is calorie-dense.", "sources": ["WebMD"], "corrective": "Eat in moderation."},
            "spicy food": {"verdict": "False", "confidence": 0.85, "explanation": "Spicy food does not cause ulcers (H. pylori bacteria does).", "sources": ["Mayo Clinic"], "corrective": "It can irritate existing ulcers."},
            "gum": {"verdict": "False", "confidence": 0.95, "explanation": "Gum does not stay in your stomach for 7 years.", "sources": ["Mayo Clinic"], "corrective": "It passes through digestion."},
            "swallowing seeds": {"verdict": "False", "confidence": 0.95, "explanation": "Watermelon seeds will not grow in your stomach.", "sources": ["Healthline"], "corrective": "They pass through."},
            "eating late": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Calories count more than the clock, but late eating can affect digestion/sleep.", "sources": ["WebMD"], "corrective": "Avoid heavy meals before bed."},
            "breakfast": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Breakfast is not strictly the 'most important meal'. Total intake matters.", "sources": ["Cleveland Clinic"], "corrective": "Eat when hungry."},
            "starvation mode": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Skipping a meal won't instantly shut down your metabolism.", "sources": ["Healthline"], "corrective": "Consistent deficit causes weight loss."},
            "negative calorie": {"verdict": "False", "confidence": 0.90, "explanation": "Celery is low calorie, but not 'negative calorie'.", "sources": ["Mayo Clinic"], "corrective": "It's just a low-cal snack."},
            "superfood": {"verdict": "Misleading", "confidence": 0.80, "explanation": "'Superfood' is a marketing term, not a medical one.", "sources": ["Harvard Health"], "corrective": "Eat a variety of foods."},
            "processed food": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Not all processed foods are bad (e.g., bagged salad, yogurt).", "sources": ["NHS"], "corrective": "Avoid ultra-processed foods."},
            "natural": {"verdict": "Misleading", "confidence": 0.70, "explanation": "'Natural' doesn't always mean safe (e.g., arsenic is natural).", "sources": ["FDA"], "corrective": "Check ingredients."},
            "chemical": {"verdict": "Misleading", "confidence": 0.70, "explanation": "Everything is made of chemicals, including water.", "sources": ["Science"], "corrective": "Don't fear the word 'chemical'."},

            # --- WEIGHT LOSS & FITNESS ---
            "spot reduction": {"verdict": "False", "confidence": 0.95, "explanation": "You cannot target fat loss in specific areas (like belly) with exercise.", "sources": ["ACE Fitness"], "corrective": "Overall weight loss reduces belly fat."},
            "crunches": {"verdict": "Misleading", "confidence": 0.90, "explanation": "Crunches build muscle but don't burn belly fat alone.", "sources": ["Mayo Clinic"], "corrective": "Combine with cardio/diet."},
            "sweat": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Sweating more doesn't mean you burned more fat, just lost water.", "sources": ["Healthline"], "corrective": "Hydrate after workout."},
            "muscle vs fat": {"verdict": "True", "confidence": 0.95, "explanation": "Muscle is denser than fat, so it takes up less space.", "sources": ["WebMD"], "corrective": "Scale weight isn't everything."},
            "no pain no gain": {"verdict": "False", "confidence": 0.90, "explanation": "Pain can indicate injury. Discomfort is okay, sharp pain is not.", "sources": ["Mayo Clinic"], "corrective": "Listen to your body."},
            "stretching": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Static stretching before a workout can reduce power. Dynamic is better.", "sources": ["Mayo Clinic"], "corrective": "Warm up dynamically."},
            "lifting weights": {"verdict": "False", "confidence": 0.90, "explanation": "Lifting weights won't make women 'bulky' without specific training/diet.", "sources": ["Womens Health"], "corrective": "Strength training is healthy."},
            "cardio": {"verdict": "True", "confidence": 0.95, "explanation": "Cardio is great for heart health.", "sources": ["AHA"], "corrective": "Aim for 150 mins/week."},
            "running knees": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Running doesn't ruin knees for everyone; it can strengthen cartilage.", "sources": ["Arthritis.org"], "corrective": "Use good form and shoes."},
            "swimming": {"verdict": "True", "confidence": 0.95, "explanation": "Swimming is a great low-impact full-body workout.", "sources": ["Harvard Health"], "corrective": "Good for joint pain."},
            "yoga": {"verdict": "True", "confidence": 0.95, "explanation": "Yoga improves flexibility and mental health.", "sources": ["NIH"], "corrective": "Start with beginner classes."},
            "pilates": {"verdict": "True", "confidence": 0.95, "explanation": "Pilates strengthens the core.", "sources": ["WebMD"], "corrective": "Good for posture."},
            "fasted cardio": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Fasted cardio doesn't burn significantly more fat long-term.", "sources": ["JISSN"], "corrective": "Do what feels best."},
            "protein window": {"verdict": "Misleading", "confidence": 0.80, "explanation": "You don't need protein immediately after a workout; total daily intake matters more.", "sources": ["JISSN"], "corrective": "Eat within a few hours."},
            "creatine": {"verdict": "True", "confidence": 0.95, "explanation": "Creatine is one of the most researched and safe supplements for performance.", "sources": ["Mayo Clinic"], "corrective": "Drink water with it."},
            "whey protein": {"verdict": "True", "confidence": 0.95, "explanation": "Whey is a convenient source of high-quality protein.", "sources": ["Healthline"], "corrective": "Food is also fine."},
            "keto": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Keto works for weight loss but can be hard to sustain and high in saturated fat.", "sources": ["Harvard Health"], "corrective": "Consult a doctor."},
            "paleo": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Paleo emphasizes whole foods but restricts healthy grains/legumes.", "sources": ["Mayo Clinic"], "corrective": "Balance is key."},
            "vegan": {"verdict": "True", "confidence": 0.90, "explanation": "A well-planned vegan diet is healthy.", "sources": ["ADA"], "corrective": "Watch B12 intake."},
            "intermittent fasting": {"verdict": "True", "confidence": 0.90, "explanation": "IF can help with weight loss and metabolic health.", "sources": ["Johns Hopkins"], "corrective": "It's not magic, just calorie control."},
            "bmi": {"verdict": "Misleading", "confidence": 0.85, "explanation": "BMI is a rough screening tool, not a perfect measure of health (misses muscle).", "sources": ["CDC"], "corrective": "Use waist circumference too."},
            "metabolism": {"verdict": "Misleading", "confidence": 0.80, "explanation": "You can't drastically 'boost' metabolism with spicy food or cold water.", "sources": ["Mayo Clinic"], "corrective": "Building muscle helps slightly."},
            "sauna": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Saunas don't melt fat, but they help relaxation and circulation.", "sources": ["Harvard Health"], "corrective": "Stay hydrated."},
            "waist trainer": {"verdict": "False", "confidence": 0.95, "explanation": "Waist trainers do not permanently reshape your body.", "sources": ["ABS"], "corrective": "Core exercises are better."},
            "toning": {"verdict": "Misleading", "confidence": 0.85, "explanation": "'Toning' is just building muscle and losing fat.", "sources": ["Womens Health"], "corrective": "Lift weights."},

            # --- DISEASES & MEDICAL ---
            "vaccin": {"verdict": "False", "confidence": 0.98, "explanation": "Vaccines do not cause autism. This myth has been debunked.", "sources": ["CDC", "WHO"], "corrective": "Vaccines are safe."},
            "autism": {"verdict": "False", "confidence": 0.98, "explanation": "Vaccines do not cause autism. This myth has been debunked.", "sources": ["CDC", "WHO"], "corrective": "Vaccines are safe."},
            "flu shot": {"verdict": "False", "confidence": 0.95, "explanation": "The flu shot cannot give you the flu (it uses dead/inactive virus).", "sources": ["CDC"], "corrective": "Get vaccinated annually."},
            "antibiotic": {"verdict": "Misleading", "confidence": 0.95, "explanation": "Antibiotics do not kill viruses (cold/flu).", "sources": ["CDC"], "corrective": "Only use for bacterial infections."},
            "antibiotic resistance": {"verdict": "True", "confidence": 0.95, "explanation": "Overuse of antibiotics leads to resistant superbugs.", "sources": ["WHO"], "corrective": "Finish your prescription."},
            "cancer cure": {"verdict": "Misleading", "confidence": 0.90, "explanation": "There is no single 'cure' for cancer. It is a complex group of diseases.", "sources": ["NCI"], "corrective": "Follow medical advice."},
            "sugar cancer": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Sugar doesn't directly feed cancer more than other cells, but obesity is a risk.", "sources": ["Mayo Clinic"], "corrective": "Maintain healthy weight."},
            "cell phones cancer": {"verdict": "Misleading", "confidence": 0.85, "explanation": "No solid evidence links cell phones to brain cancer.", "sources": ["NCI"], "corrective": "Use hands-free if worried."},
            "deodorant cancer": {"verdict": "False", "confidence": 0.90, "explanation": "Antiperspirants do not cause breast cancer.", "sources": ["Cancer.org"], "corrective": "They are safe."},
            "bra cancer": {"verdict": "False", "confidence": 0.90, "explanation": "Wearing underwire bras does not cause cancer.", "sources": ["Cancer.org"], "corrective": "Wear what fits."},
            "mammogram": {"verdict": "True", "confidence": 0.95, "explanation": "Mammograms are crucial for early breast cancer detection.", "sources": ["Cancer.org"], "corrective": "Screen regularly."},
            "diabetes sugar": {"verdict": "Misleading", "confidence": 0.90, "explanation": "Eating sugar doesn't directly cause Type 1 diabetes. Type 2 is linked to weight/lifestyle.", "sources": ["ADA"], "corrective": "Limit sugar."},
            "insulin": {"verdict": "True", "confidence": 0.95, "explanation": "Insulin is life-saving for diabetics.", "sources": ["ADA"], "corrective": "Follow doctor's orders."},
            "heart attack": {"verdict": "True", "confidence": 0.95, "explanation": "Chest pain is a common symptom, but women may have different symptoms.", "sources": ["AHA"], "corrective": "Call emergency services."},
            "stroke": {"verdict": "True", "confidence": 0.95, "explanation": "FAST: Face drooping, Arm weakness, Speech difficulty, Time to call 911.", "sources": ["Stroke.org"], "corrective": "Act fast."},
            "hypertension": {"verdict": "True", "confidence": 0.95, "explanation": "High blood pressure is the 'silent killer'.", "sources": ["AHA"], "corrective": "Check BP regularly."},
            "cholesterol meds": {"verdict": "True", "confidence": 0.90, "explanation": "Statins are effective for lowering heart risk.", "sources": ["Mayo Clinic"], "corrective": "Discuss side effects with doctor."},
            "aspirin": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Daily aspirin isn't for everyone anymore due to bleeding risk.", "sources": ["Mayo Clinic"], "corrective": "Ask your doctor."},
            "cold weather": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Cold weather doesn't cause colds; viruses do. We stay indoors more.", "sources": ["Mayo Clinic"], "corrective": "Wash hands."},
            "wet hair": {"verdict": "False", "confidence": 0.90, "explanation": "Going out with wet hair won't give you a cold.", "sources": ["Mayo Clinic"], "corrective": "You might just feel cold."},
            "chicken soup": {"verdict": "True", "confidence": 0.80, "explanation": "Chicken soup has anti-inflammatory properties and hydrates.", "sources": ["Chest Journal"], "corrective": "Good for colds."},
            "feed a cold": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Eat when hungry, stay hydrated for both colds and fevers.", "sources": ["Scientific American"], "corrective": "Listen to your body."},
            "green mucus": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Green mucus doesn't always mean bacterial infection.", "sources": ["CDC"], "corrective": "Don't demand antibiotics."},
            "allergies": {"verdict": "True", "confidence": 0.95, "explanation": "Allergies are immune overreactions.", "sources": ["AAAAI"], "corrective": "Avoid triggers."},
            "local honey": {"verdict": "Misleading", "confidence": 0.70, "explanation": "Local honey is unlikely to cure seasonal allergies.", "sources": ["WebMD"], "corrective": "Use antihistamines."},
            "poison ivy": {"verdict": "False", "confidence": 0.90, "explanation": "Poison ivy rash is not contagious. The oil is.", "sources": ["AAD"], "corrective": "Wash oil off skin."},
            "warts": {"verdict": "False", "confidence": 0.90, "explanation": "Toads do not give you warts. Viruses do.", "sources": ["WebMD"], "corrective": "Wash hands."},
            "lice": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Lice prefer clean hair, not dirty hair.", "sources": ["CDC"], "corrective": "Check regularly."},
            "shaving": {"verdict": "False", "confidence": 0.90, "explanation": "Hair does not grow back thicker after shaving.", "sources": ["Mayo Clinic"], "corrective": "It just feels coarser."},
            "grey hair": {"verdict": "False", "confidence": 0.90, "explanation": "Plucking grey hair won't make two grow back.", "sources": ["Scientific American"], "corrective": "It damages the follicle."},
            "acne": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Greasy food/chocolate isn't the main cause of acne; hormones/bacteria are.", "sources": ["AAD"], "corrective": "Keep face clean."},
            "toothpaste acne": {"verdict": "Misleading", "confidence": 0.75, "explanation": "Toothpaste can irritate acne.", "sources": ["AAD"], "corrective": "Use acne medication."},
            "pores": {"verdict": "False", "confidence": 0.90, "explanation": "Pores cannot open and close (they have no muscle).", "sources": ["Healthline"], "corrective": "Steam loosens debris."},
            "sunscreen": {"verdict": "True", "confidence": 0.95, "explanation": "Sunscreen prevents skin cancer and aging.", "sources": ["AAD"], "corrective": "Wear SPF 30+ daily."},
            "tanning": {"verdict": "Misleading", "confidence": 0.90, "explanation": "There is no 'safe' tan. It indicates skin damage.", "sources": ["SkinCancer.org"], "corrective": "Use self-tanner."},
            "vitamin d sun": {"verdict": "Misleading", "confidence": 0.85, "explanation": "You don't need to burn to get Vitamin D.", "sources": ["SkinCancer.org"], "corrective": "Supplements are safer."},
            "blue light": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Blue light from screens won't blind you, but affects sleep.", "sources": ["AAO"], "corrective": "Use night mode."},
            "reading in dark": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Reading in dim light causes eye strain, not permanent damage.", "sources": ["Mayo Clinic"], "corrective": "Use good lighting."},
            "carrots eyes": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Carrots are good for eyes but won't give you night vision.", "sources": ["Scientific American"], "corrective": "Eat a balanced diet."},
            "sitting close to tv": {"verdict": "False", "confidence": 0.90, "explanation": "Sitting close to TV won't damage eyes.", "sources": ["AAO"], "corrective": "Take breaks."},
            "glasses": {"verdict": "False", "confidence": 0.90, "explanation": "Wearing glasses doesn't make your vision worse.", "sources": ["Mayo Clinic"], "corrective": "Wear them if needed."},
            "teeth whitening": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Over-whitening can damage enamel.", "sources": ["ADA"], "corrective": "Consult a dentist."},
            "flossing": {"verdict": "True", "confidence": 0.90, "explanation": "Flossing removes plaque where brushes can't reach.", "sources": ["ADA"], "corrective": "Floss daily."},
            "root canal": {"verdict": "False", "confidence": 0.90, "explanation": "Root canals do not cause illness/cancer.", "sources": ["AAE"], "corrective": "They save teeth."},
            
            # --- MENTAL HEALTH & SLEEP ---
            "depression": {"verdict": "Misleading", "confidence": 0.90, "explanation": "Depression is not just 'being sad'. It's a clinical condition.", "sources": ["NIMH"], "corrective": "Seek professional help."},
            "anxiety": {"verdict": "True", "confidence": 0.90, "explanation": "Anxiety disorders are real medical conditions.", "sources": ["NIMH"], "corrective": "Therapy helps."},
            "adhd": {"verdict": "Misleading", "confidence": 0.85, "explanation": "ADHD is not caused by bad parenting or sugar.", "sources": ["CDC"], "corrective": "It's neurological."},
            "bipolar": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Mood swings don't automatically mean Bipolar Disorder.", "sources": ["NIMH"], "corrective": "Requires diagnosis."},
            "ocd": {"verdict": "Misleading", "confidence": 0.85, "explanation": "OCD is not just liking things tidy. It involves intrusive thoughts.", "sources": ["NIMH"], "corrective": "Seek help."},
            "schizophrenia": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Schizophrenia usually doesn't mean 'multiple personalities'.", "sources": ["NIMH"], "corrective": "It involves psychosis."},
            "brain usage": {"verdict": "False", "confidence": 0.90, "explanation": "We use more than 10% of our brains.", "sources": ["Scientific American"], "corrective": "The whole brain is active."},
            "left brain": {"verdict": "Misleading", "confidence": 0.80, "explanation": "People aren't strictly 'left-brained' or 'right-brained'.", "sources": ["Healthline"], "corrective": "Brain works together."},
            "sleep": {"verdict": "Misleading", "confidence": 0.85, "explanation": "You cannot 'catch up' on sleep fully on weekends.", "sources": ["Sleep Foundation"], "corrective": "Maintain consistent sleep schedule."},
            "8 hours": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Sleep needs vary (7-9 hours is typical).", "sources": ["Sleep Foundation"], "corrective": "Find your number."},
            "snoring": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Loud snoring can be a sign of sleep apnea.", "sources": ["Mayo Clinic"], "corrective": "See a doctor."},
            "insomnia": {"verdict": "True", "confidence": 0.90, "explanation": "CBT-I is effective for insomnia.", "sources": ["Sleep Foundation"], "corrective": "Improve sleep hygiene."},
            "dreams": {"verdict": "Misleading", "confidence": 0.70, "explanation": "Dreams don't predict the future.", "sources": ["Psychology Today"], "corrective": "They process emotions."},
            "turkey": {"verdict": "False", "confidence": 0.90, "explanation": "Turkey doesn't make you sleepy (tryptophan is low). Overeating does.", "sources": ["WebMD"], "corrective": "Eat moderate portions."},
            "warm milk": {"verdict": "Misleading", "confidence": 0.75, "explanation": "Warm milk may be psychological comfort, not chemical.", "sources": ["Sleep Foundation"], "corrective": "Relaxing routine helps."},
            "alcohol sleep": {"verdict": "Misleading", "confidence": 0.90, "explanation": "Alcohol helps you fall asleep but ruins sleep quality.", "sources": ["Sleep Foundation"], "corrective": "Avoid before bed."},
            "screens sleep": {"verdict": "True", "confidence": 0.95, "explanation": "Blue light suppresses melatonin.", "sources": ["Sleep Foundation"], "corrective": "Stop screens 1hr before bed."},

            # --- COVID & 5G & CONSPIRACIES ---
            "bleach": {"verdict": "False", "confidence": 0.99, "explanation": "Drinking bleach is dangerous and does not cure COVID-19.", "sources": ["WHO", "CDC"], "corrective": "Do not ingest disinfectants."},
            "microchip": {"verdict": "False", "confidence": 0.99, "explanation": "Vaccines do not contain microchips.", "sources": ["Reuters", "BBC"], "corrective": "Vaccines contain biological ingredients to build immunity."},
            "magnet": {"verdict": "False", "confidence": 0.99, "explanation": "Vaccines do not make you magnetic.", "sources": ["CDC"], "corrective": "None."},
            "ivermectin": {"verdict": "Misleading", "confidence": 0.90, "explanation": "Ivermectin is not approved for treating COVID-19.", "sources": ["FDA", "WHO"], "corrective": "Use approved treatments."},
            "mask": {"verdict": "True", "confidence": 0.95, "explanation": "Masks reduce the spread of respiratory viruses.", "sources": ["CDC", "Nature"], "corrective": "Wear masks in high-risk areas."},
            "5g": {"verdict": "False", "confidence": 0.99, "explanation": "5G does not spread viruses.", "sources": ["WHO", "FCC"], "corrective": "Viruses spread via droplets."},
            "chemtrails": {"verdict": "False", "confidence": 0.95, "explanation": "Contrails are water vapor, not chemicals.", "sources": ["EPA"], "corrective": "It's condensation."},
            "flat earth": {"verdict": "False", "confidence": 0.99, "explanation": "The Earth is round.", "sources": ["NASA"], "corrective": "Physics."},
            "moon landing": {"verdict": "True", "confidence": 0.99, "explanation": "Humans landed on the moon.", "sources": ["NASA"], "corrective": "History."},

            # --- FIRST AID & SAFETY ---
            "butter burn": {"verdict": "False", "confidence": 0.95, "explanation": "Do not put butter on burns; it traps heat.", "sources": ["Mayo Clinic"], "corrective": "Use cool water."},
            "ice burn": {"verdict": "Misleading", "confidence": 0.90, "explanation": "Don't put ice directly on severe burns.", "sources": ["Mayo Clinic"], "corrective": "Cool water is best."},
            "nosebleed": {"verdict": "Misleading", "confidence": 0.95, "explanation": "Don't tilt head back for nosebleed (blood goes to stomach).", "sources": ["Mayo Clinic"], "corrective": "Lean forward and pinch."},
            "venom": {"verdict": "False", "confidence": 0.95, "explanation": "Do not suck out snake venom.", "sources": ["CDC"], "corrective": "Seek emergency help."},
            "seizure": {"verdict": "False", "confidence": 0.95, "explanation": "Do not put anything in a seizure patient's mouth.", "sources": ["CDC"], "corrective": "Keep them safe, time it."},
            "cpr": {"verdict": "True", "confidence": 0.95, "explanation": "CPR saves lives.", "sources": ["AHA"], "corrective": "Learn Hands-Only CPR."},
            "heimlich": {"verdict": "True", "confidence": 0.95, "explanation": "Heimlich maneuver clears choking.", "sources": ["Mayo Clinic"], "corrective": "Learn it."},
            "drowning": {"verdict": "Misleading", "confidence": 0.90, "explanation": "Drowning is often silent, not splashing.", "sources": ["CDC"], "corrective": "Watch kids closely."},
            "concussion": {"verdict": "Misleading", "confidence": 0.85, "explanation": "You don't always need to stay awake after a concussion, but check with doctor.", "sources": ["Mayo Clinic"], "corrective": "Rest is needed."},
            "hydrogen peroxide": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Peroxide can damage tissue. Water/soap is better for cuts.", "sources": ["WebMD"], "corrective": "Clean with water."},
            "rubbing alcohol": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Alcohol stings and damages tissue.", "sources": ["WebMD"], "corrective": "Use water/soap."},
            
            # --- WOMEN'S HEALTH ---
            "period sync": {"verdict": "Misleading", "confidence": 0.70, "explanation": "Period syncing is likely a myth/coincidence.", "sources": ["Cleveland Clinic"], "corrective": "Cycles vary."},
            "pms": {"verdict": "True", "confidence": 0.95, "explanation": "PMS is real.", "sources": ["Mayo Clinic"], "corrective": "Manage symptoms."},
            "cranberry juice": {"verdict": "Misleading", "confidence": 0.80, "explanation": "Cranberry juice may help prevent UTIs but doesn't cure them.", "sources": ["WebMD"], "corrective": "See a doctor for antibiotics."},
            "yeast infection": {"verdict": "True", "confidence": 0.95, "explanation": "Yeast infections are common.", "sources": ["Mayo Clinic"], "corrective": "Treatable."},
            "birth control": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Birth control doesn't cause infertility long-term.", "sources": ["Planned Parenthood"], "corrective": "Consult doctor."},
            "morning sickness": {"verdict": "Misleading", "confidence": 0.90, "explanation": "Morning sickness can happen all day.", "sources": ["Mayo Clinic"], "corrective": "Small meals help."},
            "eating for two": {"verdict": "Misleading", "confidence": 0.90, "explanation": "Pregnant women only need ~300 extra calories.", "sources": ["ACOG"], "corrective": "Nutrient density matters."},
            
            # --- MEN'S HEALTH ---
            "prostate": {"verdict": "True", "confidence": 0.95, "explanation": "Prostate health is important for older men.", "sources": ["Cancer.org"], "corrective": "Screening helps."},
            "testosterone": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Supplements don't always boost T safely.", "sources": ["Mayo Clinic"], "corrective": "Lifestyle matters."},
            "baldness": {"verdict": "Misleading", "confidence": 0.85, "explanation": "Baldness comes from both parents, not just mother's side.", "sources": ["Healthline"], "corrective": "Genetics."},
            "hats": {"verdict": "False", "confidence": 0.90, "explanation": "Wearing hats doesn't cause baldness.", "sources": ["Mayo Clinic"], "corrective": "It's genetic."},
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
            "explanation": f"The claim '{query}' requires further investigation. In this demo version (Mock Mode), I recognize 100+ common health myths. Try asking about 'sugar', 'detox', '5G', 'vaccines', 'sleep', 'crunches', 'butter on burns', etc.",
            "sources": [],
            "corrective_information": "Please consult a medical professional."
        }
