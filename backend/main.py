from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models import AnalysisRequest, AnalysisResult
from agent import get_agent, extract_text_from_image

app = FastAPI(title="Veritas Health Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Veritas Health Agent API is running"}

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_claim(request: AnalysisRequest):
    agent = get_agent()
    
    query = request.text
    if request.image_base64:
        print("Processing image with Gemini Vision...")
        extracted_text = extract_text_from_image(request.image_base64)
        print(f"Extracted Text: {extracted_text[:100]}...")
        
        if query:
            query = f"{query}\n\nContext from Image: {extracted_text}"
        else:
            query = f"Analyze this text found in an image: {extracted_text}"
    
    if not query:
        raise HTTPException(status_code=400, detail="No text or image provided")

    try:
        # Run the agent
        # The agent now returns a dict (either from JSON parse or mock)
        result = agent.invoke({"input": f"Analyze this health claim for misinformation: {query}. Return the verdict (True/False/Misleading), confidence, explanation, sources, and corrective info."})
        
        return AnalysisResult(
            verdict=result.get("verdict", "Unverified"),
            confidence=result.get("confidence", 0.0),
            explanation=result.get("explanation", "No explanation provided."),
            sources=result.get("sources", []),
            corrective_information=result.get("corrective_information", None)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
