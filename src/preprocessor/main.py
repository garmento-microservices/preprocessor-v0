from fastapi import FastAPI

from .module import provide_injector

injector = provide_injector()
app = injector.get(FastAPI)
