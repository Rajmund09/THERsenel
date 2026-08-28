"""
Intrusion detection logic module for ROI polygon crossing, multi-object tracking, and alert dispatching.
"""

from src.intrusion.border_tracker import BorderIntrusionTracker
from src.intrusion.roi import ROIManager
from src.intrusion.tracker import SimpleObjectTracker
from src.intrusion.alert import IntrusionAlertDispatcher

__all__ = [
    "BorderIntrusionTracker",
    "ROIManager",
    "SimpleObjectTracker",
    "IntrusionAlertDispatcher",
]
