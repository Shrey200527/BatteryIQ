"""
BatteryIQ Dashboard Backend - Main Application.

FastAPI application that provides REST API endpoints for battery health predictions,
integrating with the existing Phase 2 ML inference interfaces.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from dashboard.backend.routes import batteries_router, predictions_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    print("🚀 BatteryIQ Dashboard Backend starting up...")
    
    # Initialize services (lazy loading will handle actual data loading)
    try:
        from dashboard.backend.services import get_battery_service
        battery_service = get_battery_service()
        print(f"✅ Battery service initialized with {len(battery_service.get_available_batteries())} batteries")
    except Exception as e:
        print(f"⚠️ Warning: Battery service initialization failed: {e}")
        print("   Some endpoints may not function correctly.")
    
    yield
    
    # Shutdown
    print("🛑 BatteryIQ Dashboard Backend shutting down...")


# Create FastAPI application
app = FastAPI(
    title="BatteryIQ Dashboard Backend",
    description="REST API for BatteryIQ battery health predictions and analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(batteries_router)
app.include_router(predictions_router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "BatteryIQ Dashboard Backend",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "batteries": "/api/batteries",
            "predictions": "/api/predict",
            "documentation": "/docs"
        }
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        from dashboard.backend.services import get_battery_service
        battery_service = get_battery_service()
        batteries = battery_service.get_available_batteries()
        
        return {
            "status": "healthy",
            "service": "BatteryIQ Dashboard Backend",
            "batteries_available": len(batteries),
            "battery_ids": batteries
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "error": str(e)
            }
        )


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    """Handle ValueError exceptions."""
    return {
        "error": "validation_error",
        "message": str(exc),
        "status_code": 400
    }


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc: FileNotFoundError):
    """Handle FileNotFoundError exceptions."""
    return {
        "error": "resource_not_found",
        "message": str(exc),
        "status_code": 404
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "dashboard.backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )