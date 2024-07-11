from __future__ import annotations

import asyncio
from functools import lru_cache
from io import UnsupportedOperation
import logging
import os
from dataclasses import dataclass
from typing import BinaryIO
from uuid import UUID, uuid4

from preprocessor.services.pose_extractor import render_posenet

from .garment_extractor.commands import extract_garment_mask
from .human_segmentator import do_human_segmentation_inference
from .job_repository import JobRepository
from .jobs import PreprocessingJob
from .pose_extractor import extract_poses, run_densepose


class PresetNotFound(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__("Preset not found", *args)


@dataclass
class PreprocessingService:
    repository: JobRepository
    BASE_FOLDER = os.getenv("BASE_FOLDER", "")
    IMAGE_SIZE = (768, 1024)

    def __hash__(self) -> int:
        return id(self)

    async def process_garment(self, job: PreprocessingJob, base_folder: str):
        # process garment
        garment_mask_output_file = os.path.join(base_folder, "garment_mask.jpg")
        garment_only_output_file = os.path.join(base_folder, "garment_only.jpg")
        extract_garment_mask(
            input_image=job.garment_image,
            output_mask_image=garment_mask_output_file,
            output_garment_only_image=garment_only_output_file,
        )
        return garment_only_output_file

    async def process_poses(self, job: PreprocessingJob, base_folder: str):
        # pose keypoints
        pose_output_file = os.path.join(base_folder, "keypoints.json")
        extract_poses(input_file=job.ref_image, output_file=pose_output_file)
        return pose_output_file

    async def process_densepose(self, job: PreprocessingJob, base_folder: str):
        # USE rendered openpose instead of densepose to temporarily make VITON-HD work
        densepose_output_file = os.path.join(base_folder, "densepose.jpg")
        render_posenet(
            input_file=job.ref_image,
            output_file=densepose_output_file,
            size=self.IMAGE_SIZE,
        )
        return densepose_output_file

    async def process_segmentation(self, job: PreprocessingJob, base_folder: str):
        # segmentation
        segmentation_output_file = os.path.join(base_folder, "segmented.jpg")
        do_human_segmentation_inference(
            img_path=job.ref_image,
            output_file=segmentation_output_file,
        )
        return segmentation_output_file

    async def process(self, job_id: str, with_preset=False):
        print(f"--- Processing job ID: {job_id} ---")
        job = self.repository.find_by_id(UUID(job_id))
        if not job:
            raise ValueError(f"Job ID {job_id} not found")

        job.processing()
        self.repository.save(job)
        try:
            base_folder = os.path.join(self.BASE_FOLDER, job_id)
            if not with_preset:
                (
                    garment_only_output_file,
                    pose_output_file,
                    densepose_output_file,
                    segmentation_output_file,
                ) = await asyncio.gather(
                    self.process_garment(job, base_folder),
                    self.process_poses(job, base_folder),
                    self.process_densepose(job, base_folder),
                    self.process_segmentation(job, base_folder),
                )

                job.success_with(
                    masked_garment_image=garment_only_output_file,
                    densepose_image=densepose_output_file,
                    segmented_image=segmentation_output_file,
                    pose_keypoints=pose_output_file,
                )
            else:
                garment_only_output_file = await self.process_garment(job, base_folder)
                job.success_with(masked_garment_image=garment_only_output_file)
            self.repository.save(job)
            print(f"--- [DONE] Processed job ID: {job_id} ---")
        except Exception as e:
            logging.error(f"Error while processing job ID {job_id}")
            logging.exception(e)
            job.failed()
            self.repository.save(job)

    def create_job(self, ref_image: BinaryIO, garment_image: BinaryIO):
        """
        Returns the job for convenience scheduling outside this. \
        Might need to improve so that a dedicated scheduler also works.
        """
        job_id = uuid4()
        base_folder = os.path.join(self.BASE_FOLDER, str(job_id))
        os.makedirs(base_folder)
        ref_image_file = os.path.join(base_folder, "ref.jpg")
        garment_image_file = os.path.join(base_folder, "garment.jpg")
        with open(ref_image_file, "wb") as file:
            while chunk := ref_image.read(1024):
                file.write(chunk)
        with open(garment_image_file, "wb") as file:
            while chunk := garment_image.read(1024):
                file.write(chunk)
        job = PreprocessingJob(
            ref_image=ref_image_file, garment_image=garment_image_file, id=job_id
        )
        self.repository.save(job)

        async def do_process():
            await self.process(str(job_id))

        return str(job.id), do_process

    def create_job_with_preset(self, preset: str, garment_image: BinaryIO):
        """
        Returns the job for convenience scheduling outside this. \
        Might need to improve so that a dedicated scheduler also works.
        """
        if preset not in self.list_presets():
            raise PresetNotFound(preset)
        preset_folder = f"presets/{preset}"
        job_id = uuid4()
        base_folder = os.path.join(self.BASE_FOLDER, str(job_id))
        os.makedirs(base_folder)
        garment_image_file = os.path.join(base_folder, "garment.jpg")
        with open(garment_image_file, "wb") as file:
            while chunk := garment_image.read(1024):
                file.write(chunk)
        ref_image_file = os.path.join(preset_folder, "ref.jpg")
        densepose_image=os.path.join(preset_folder, "densepose.jpg")
        pose_keypoints=os.path.join(preset_folder, "keypoints.json")
        segmented_image=os.path.join(preset_folder, "segmented.jpg")
        job = PreprocessingJob(
            id=job_id,
            ref_image=ref_image_file,
            garment_image=garment_image_file,
            densepose_image=densepose_image,
            pose_keypoints=pose_keypoints,
            segmented_image=segmented_image,
        )
        self.repository.save(job)

        async def do_process():
            await self.process(str(job_id), with_preset=True)

        return str(job.id), do_process

    def get_job(self, job_id: str):
        return self.repository.find_by_id(UUID(job_id))

    def abort_job(self, job_id: str):
        raise UnsupportedOperation()
        # job = self.repository.find_by_id(UUID(job_id))
        # job.aborted()
        # self.repository.save(job)

    def to_preset_meta(self, preset: str):
        return {
            "name": preset,
            "refImage": f"/presets/{preset}/ref.jpg",
            "denseposeImage": f"/presets/{preset}/densepose.jpg",
            "segmented": f"/presets/{preset}/segmented.jpg",
            "poseKeypoints": f"/presets/{preset}/keypoints.json",
        }

    @lru_cache
    def list_presets(self, root_dir="presets") -> dict[str, dict[str, str]]:
        presets = [
            maybe_preset.removeprefix(f"{root_dir}/")
            for maybe_preset, _, preset_files in os.walk(root_dir)
            if "ref.jpg" in preset_files
        ]
        return {preset: self.to_preset_meta(preset) for preset in presets}
    
    def get_preset_ref_image(self, preset_name: str, root_dir="presets"):
        if preset_name not in self.list_presets():
            raise PresetNotFound(preset_name)
        return f"{root_dir}/{preset_name}/ref.jpg"
