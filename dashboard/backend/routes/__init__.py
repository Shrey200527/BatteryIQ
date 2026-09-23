"""
Routes module for BatteryIQ Dashboard Backend.
"""

from .batteries import router as batteries_router
from .predictions import router as predictions_router

__all__ = [
    'batteries_router',
    'predictions_router',
]