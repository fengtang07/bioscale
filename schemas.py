from pydantic import BaseModel, Field
from typing import List, Dict

class ProtocolStep(BaseModel):
    """Defines a single step in a protocol."""
    step_number: int = Field(..., description="The sequential number of the step.")
    description: str = Field(..., description="Detailed description of the action to be performed.")
    duration_minutes: int = Field(0, description="Estimated time in minutes to complete the step. 0 if not applicable.")

class Protocol(BaseModel):
    """Defines the structure for an experimental protocol."""
    title: str = Field(..., description="The official title of the protocol.")
    objective: str = Field(..., description="A brief, one-sentence description of the protocol's goal.")
    materials: Dict[str, str] = Field(..., description="A dictionary of required materials, reagents, and equipment.")
    steps: List[ProtocolStep] = Field(..., description="A list of sequential steps to follow.")