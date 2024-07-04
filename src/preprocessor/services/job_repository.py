from typing import Protocol, runtime_checkable
from uuid import UUID

from .jobs import PreprocessingJob


class NotFound(Exception):
    def __init__(self, _type: type, *args) -> None:
        super().__init__(f"{_type.__name__} instance is not found with filter(s)", *args)


@runtime_checkable
class JobRepository(Protocol):
    def find_by_id(self, id: UUID) -> PreprocessingJob: ...
    def save(self, job: PreprocessingJob): ...
