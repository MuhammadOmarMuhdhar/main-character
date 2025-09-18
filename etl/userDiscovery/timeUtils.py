import logging
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)


def is_update_time() -> bool:
    """
    Check if current time is 2 AM Pacific Time
    
    Returns:
        True if current time is 2 AM Pacific, False otherwise
    """
    try:
        # Get current UTC time
        utc_now = datetime.utcnow().replace(tzinfo=pytz.UTC)
        
        # Convert to Pacific Time
        pacific_tz = pytz.timezone('US/Pacific')
        pacific_now = utc_now.astimezone(pacific_tz)
        
        # Check if it's 2 AM Pacific (hour == 2)
        return pacific_now.hour == 2
        
    except Exception as e:
        logger.error(f"Failed to check update time: {e}")
        return False