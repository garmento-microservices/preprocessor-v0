from pydantic import BaseModel


class JobResponse(BaseModel):
    id: str
    refImage: str | None = None  # URL
    garmentImage: str | None = None  # URL
    maskedGarmentImage: str | None = None  # cloth_mask
    denseposeImage: str | None = None  # image-densepose
    segmentedImage: str | None = None  # image-parse-v3
    poseKeypoints: str | None = None  # openpose_json
