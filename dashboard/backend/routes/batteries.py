"""
Battery Routes for BatteryIQ Dashboard Backend.

These routes handle battery-related endpoints, providing information
about available batteries and their data.
"""

from fastapi import APIRouter, HTTPException, status
from typing import List

from dashboard.backend.schemas import (
    BatteryInfo,
    BatteriesResponse,
    BatteryDetail,
    ErrorResponse
)
from dashboard.backend.services import get_battery_service

router = APIRouter(prefix="/api/batteries", tags=["batteries"])


@router.get("", response_model=BatteriesResponse)
async def get_batteries():
    """
    Get list of all available batteries.
    
    Returns information about each battery including total cycles
    and RUL validity status.
    """
    try:
        battery_service = get_battery_service()
        available_batteries = battery_service.get_available_batteries()
        
        batteries_info = []
        for battery_id in available_batteries:
            info = battery_service.get_battery_info(battery_id)
            batteries_info.append(BatteryInfo(**info))
        
        return BatteriesResponse(
            batteries=batteries_info,
            total=len(batteries_info)
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "data_unavailable",
                "message": "Battery data files not found",
                "details": str(e)
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_error",
                "message": "Failed to retrieve battery information",
                "details": str(e)
            }
        )


@router.get("/{battery_id}", response_model=BatteryDetail)
async def get_battery_detail(battery_id: str):
    """
    Get detailed information about a specific battery.
    
    Includes SOH range, available cycles, and RUL validity.
    """
    try:
        battery_service = get_battery_service()
        
        if not battery_service.validate_battery_id(battery_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "invalid_battery",
                    "message": f"Battery {battery_id} is not available"
                }
            )
        
        detail = battery_service.get_battery_detail(battery_id)
        return BatteryDetail(**detail)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_error",
                "message": f"Failed to retrieve details for battery {battery_id}",
                "details": str(e)
            }
        )


@router.get("/{battery_id}/history")
async def get_battery_history(battery_id: str):
    """
    Get historical data for a specific battery.
    
    Returns available cycles and SOH progression.
    """
    try:
        battery_service = get_battery_service()
        
        if not battery_service.validate_battery_id(battery_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "invalid_battery",
                    "message": f"Battery {battery_id} is not available"
                }
            )
        
        detail = battery_service.get_battery_detail(battery_id)
        
        return {
            "battery_id": battery_id,
            "total_cycles": detail["total_cycles"],
            "available_cycles": detail["available_cycles"],
            "soh_range": detail["soh_range"],
            "is_rul_valid": detail["is_rul_valid"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_error",
                "message": f"Failed to retrieve history for battery {battery_id}",
                "details": str(e)
            }
        )