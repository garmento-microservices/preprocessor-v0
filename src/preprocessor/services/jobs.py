from enum import Enum
from uuid import UUID, uuid4

from pydantic.dataclasses import dataclass


class JobStatus(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


class NotProcessing(Exception): ...


class AlreadyProcessing(Exception): ...


class AlreadyAborted(Exception): ...


@dataclass
class PreprocessingJob:
    ref_image: str  # image
    garment_image: str  # cloth

    id: UUID = uuid4()
    status: JobStatus = JobStatus.PENDING
    # store filenames
    masked_garment_image: str | None = None  # cloth_mask
    densepose_image: str | None = None  # image-densepose
    segmented_image: str | None = None  # image-parse-v3
    pose_keypoints: str | None = None  # openpose_json

    def processing(self):
        if self.status != JobStatus.PENDING:
            raise AlreadyProcessing()
        self.status = JobStatus.IN_PROGRESS

    def success_with(
        self,
        masked_garment_image: str,
        densepose_image: str | None = None,
        segmented_image: str | None = None,
        pose_keypoints: str | None = None,
    ):
        if self.status not in (JobStatus.IN_PROGRESS,):
            raise NotProcessing()
        not_success = (
            (not densepose_image and not self.densepose_image)
            or (not segmented_image and not self.segmented_image)
            or (not pose_keypoints and not self.pose_keypoints)
        )
        if not_success:
            self.failed()
        else:
            self.masked_garment_image = masked_garment_image
            self.densepose_image = densepose_image or self.densepose_image
            self.segmented_image = segmented_image or self.segmented_image
            self.pose_keypoints = pose_keypoints or self.pose_keypoints
            self.status = JobStatus.SUCCESS

    def failed(self):
        if self.status not in (JobStatus.IN_PROGRESS,):
            raise NotProcessing()
        self.status = JobStatus.FAILED

    def aborted(self):
        if self.status not in (JobStatus.IN_PROGRESS,):
            raise NotProcessing()
        self.status = JobStatus.ABORTED
