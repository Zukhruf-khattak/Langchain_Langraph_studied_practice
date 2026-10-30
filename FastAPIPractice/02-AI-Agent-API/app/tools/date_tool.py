from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo


def get_current_datetime(timezone_name: str = "UTC") -> str:
    """
    Get the current date and time in a specified timezone.

    Args:
        timezone_name: Timezone string name, e.g. 'UTC', 'America/New_York', 'Asia/Kolkata', 'Europe/London'. Defaults to 'UTC'.

    Returns:
        Formatted current date, time, weekday, and timezone offset.
    """
    try:
        if timezone_name.upper() == "UTC":
            tz = timezone.utc
        else:
            tz = ZoneInfo(timezone_name)
        now = datetime.now(tz)
        return (
            f"Current DateTime: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            f"ISO Format: {now.isoformat()}\n"
            f"Day of the week: {now.strftime('%A')}"
        )
    except Exception as e:
        return f"Error retrieving datetime for timezone '{timezone_name}': {str(e)}"


def calculate_date_offset(days: int = 0, hours: int = 0, timezone_name: str = "UTC") -> str:
    """
    Calculate a future or past date/time offset from now.

    Args:
        days: Number of days to add (positive) or subtract (negative). Defaults to 0.
        hours: Number of hours to add (positive) or subtract (negative). Defaults to 0.
        timezone_name: Timezone string name (e.g. 'UTC', 'America/New_York'). Defaults to 'UTC'.

    Returns:
        The target calculated date and time string.
    """
    try:
        if timezone_name.upper() == "UTC":
            tz = timezone.utc
        else:
            tz = ZoneInfo(timezone_name)
        now = datetime.now(tz)
        target = now + timedelta(days=days, hours=hours)
        return (
            f"Target DateTime: {target.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            f"Offset: {days} days, {hours} hours\n"
            f"Day of the week: {target.strftime('%A')}"
        )
    except Exception as e:
        return f"Error calculating offset: {str(e)}"


def get_date_difference(start_date: str, end_date: str) -> str:
    """
    Calculate the difference in days and hours between two dates (format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).

    Args:
        start_date: Starting date string (e.g. '2026-01-01' or '2026-01-01 12:00:00').
        end_date: Ending date string (e.g. '2026-12-31' or '2026-12-31 23:59:59').

    Returns:
        The difference description between the two dates.
    """
    formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"]

    def parse_dt(d_str: str) -> datetime:
        d_str = d_str.strip()
        for fmt in formats:
            try:
                return datetime.strptime(d_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Unable to parse date '{d_str}'. Supported formats include YYYY-MM-DD and YYYY-MM-DD HH:MM:SS.")

    try:
        dt1 = parse_dt(start_date)
        dt2 = parse_dt(end_date)
        diff = dt2 - dt1
        total_seconds = int(diff.total_seconds())
        days = diff.days
        hours = (total_seconds % (24 * 3600)) // 3600

        return (
            f"Difference from {start_date} to {end_date}:\n"
            f"- Total Days: {days}\n"
            f"- Extra Hours: {hours}\n"
            f"- Total Hours: {total_seconds // 3600}\n"
            f"- Direction: {'Future' if diff.total_seconds() >= 0 else 'Past'}"
        )
    except Exception as e:
        return f"Error calculating date difference: {str(e)}"
