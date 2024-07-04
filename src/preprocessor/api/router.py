import os
from typing import Annotated, Literal
from uuid import UUID

from fastapi import BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.routing import APIRouter
from injector import inject

from ..services.job_repository import NotFound
from ..services.preprocessing_service import PreprocessingService
from .responses import JobResponse

PresetImageFileName = Literal[
    "ref.jpg",
    "densepose.jpg",
    "segmented.jpg",
    "keypoints.json",
]


@inject
class PreprocessingRouter:
    def __init__(self, service: PreprocessingService) -> None:
        self.service = service
        self.router = APIRouter()
        self.router.post("/jobs")(self.create_job)
        self.router.get("/jobs/{job_id}")(self.get_job_status)
        self.router.delete("/jobs/{job_id}")(self.abort_job)
        self.router.get("/presets")(self.list_presets)
        self.router.get("/presets/{preset}")(self.get_preset_meta)
        self.router.get("/presets/{preset}/{name}")(self.serve_preset_image)

    def serve_preset_image(self, preset: str, name: PresetImageFileName):
        filename = os.path.join("presets", preset, name)
        if not os.path.exists(filename):
            raise HTTPException(404, detail="Not Found")
        return FileResponse(filename)

    def list_presets(self):
        return list(self.service.list_presets().values())

    def get_preset_meta(self, preset: str):
        presets = self.service.list_presets()
        if preset not in presets:
            raise HTTPException(404, f"preset not found: {preset}")
        return presets[preset]

    def create_job(
        self,
        garment_image: Annotated[UploadFile, File()],
        background_tasks: BackgroundTasks,
        preset_id: Annotated[str | None, Form()] = None,
        ref_image: Annotated[UploadFile | None, File()] = None,
    ) -> JobResponse:
        if preset_id is None and ref_image is None:
            raise HTTPException(400)
        if preset_id is None:
            assert ref_image is not None
            job_id, background_task = self.service.create_job(
                ref_image=ref_image.file, garment_image=garment_image.file
            )
        else:
            job_id, background_task = self.service.create_job_with_preset(
                preset=preset_id, garment_image=garment_image.file
            )
        background_tasks.add_task(background_task)
        return JobResponse(id=job_id)

    def get_job_status(self, job_id: UUID) -> JobResponse:
        try:
            job = self.service.get_job(str(job_id))
            return JobResponse(
                id=str(job.id),
                refImage=job.ref_image,
                garmentImage=job.garment_image,
                maskedGarmentImage=job.masked_garment_image,
                denseposeImage=job.densepose_image,
                segmentedImage=job.segmented_image,
                poseKeypoints=job.pose_keypoints,
            )
        except NotFound as e:
            raise HTTPException(404, ": ".join(str(arg) for arg in e.args))

    def abort_job(self, job_id: str):
        self.service.abort_job(job_id)
        return {}
