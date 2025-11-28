from pydantic import BaseModel
from typing import List, Optional

class AnalysisRequest(BaseModel):
    text: Optional[str] = None
    image_base64: Optional[str] = None

class AnalysisResult(BaseModel):
    verdict: str  # "True", "False", "Misleading", "Unverified"
    confidence: float
    explanation: str
    sources: List[str]
    corrective_information: str
