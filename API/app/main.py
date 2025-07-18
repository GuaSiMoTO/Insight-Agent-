from fastapi import FastAPI
import API.app.router as router

app = FastAPI(title="API para enviar TXT a n8n")

app.include_router(router, prefix="/api")