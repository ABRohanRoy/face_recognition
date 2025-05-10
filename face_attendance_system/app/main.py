# File: app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import endpoints

# Create FastAPI app
app = FastAPI(
    title="Face Recognition Attendance System",
    description="A FastAPI backend for facial recognition based attendance tracking",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(endpoints.router, prefix="/api")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Face Recognition Attendance System API",
        "docs_url": "/docs",
        "version": "1.0.0"
    }