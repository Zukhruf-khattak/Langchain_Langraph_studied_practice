from app.tools.calculator import calculate
from app.tools.date_tool import (
    calculate_date_offset,
    get_current_datetime,
    get_date_difference,
)
from app.tools.text_analyzer import analyze_text

# Registered tool functions available to the Agent
AVAILABLE_TOOLS = [
    calculate,
    get_current_datetime,
    calculate_date_offset,
    get_date_difference,
    analyze_text,
]

__all__ = [
    "calculate",
    "get_current_datetime",
    "calculate_date_offset",
    "get_date_difference",
    "analyze_text",
    "AVAILABLE_TOOLS",
]
