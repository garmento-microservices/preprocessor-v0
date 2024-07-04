from uuid import UUID

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from ..services.jobs import JobStatus, PreprocessingJob
from .base import Base


class JobRecord(Base):
    __tablename__ = "preprocessing_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(30))
    ref_image: Mapped[str] = mapped_column(String(1024))
    garment_image: Mapped[str] = mapped_column(String(1024))
    masked_garment_image: Mapped[str] = mapped_column(String(1024), nullable=True)
    densepose_image: Mapped[str] = mapped_column(String(1024), nullable=True)
    segmented_image: Mapped[str] = mapped_column(String(1024), nullable=True)
    pose_keypoints: Mapped[str] = mapped_column(String(1024), nullable=True)

    def to_domain(self):
        status = self.status

        return PreprocessingJob(
            id=UUID(self.id),
            status=JobStatus(status),
            ref_image=self.ref_image,
            garment_image=self.garment_image,
            masked_garment_image=self.masked_garment_image,
            densepose_image=self.densepose_image,
            segmented_image=self.segmented_image,
            pose_keypoints=self.pose_keypoints,
        )

    @staticmethod
    def from_domain(job: PreprocessingJob):
        return JobRecord(
            id=str(job.id),
            status=job.status.value,
            ref_image=job.ref_image,
            garment_image=job.garment_image,
            masked_garment_image=job.masked_garment_image,
            densepose_image=job.densepose_image,
            segmented_image=job.segmented_image,
            pose_keypoints=job.pose_keypoints,
        )
