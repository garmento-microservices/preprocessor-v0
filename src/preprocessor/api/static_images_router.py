import os
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

ImageFileName = Literal[
    "ref.jpg",
    "garment.jpg",
    "garment_only.jpg",
    "densepose.jpg",
    "segmented.jpg",
    "keypoints.json",
]


class ImagesRouter:
    def __init__(self) -> None:
        self.router = APIRouter(prefix="/images")
        self.router.get("/{job_id}/{name}")(self.serve_image)

    def serve_image(self, job_id: str, name: ImageFileName):
        filename = os.path.join("images", job_id, name)
        if not os.path.exists(filename):
            raise HTTPException(404, detail="Not Found")
        return FileResponse(filename)
