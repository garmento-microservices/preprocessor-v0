from pydantic import BaseModel


class CreateJobRequest(BaseModel):
    ref_image: str  # base64 string
    garment_image: str  # base64 string
