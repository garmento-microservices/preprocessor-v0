from fastapi import APIRouter


class HealthCheckRouter:
    def __init__(self) -> None:
        self.router = APIRouter(prefix="/health")
        self.router.get("")(self.do_health_check)
    
    def do_health_check(self):
        return {"status": "UP"}
