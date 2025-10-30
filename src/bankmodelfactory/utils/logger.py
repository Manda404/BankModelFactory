from loguru import logger
import sys
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Remove default handler to avoid duplicate logs
logger.remove()

# Console output (human-readable)
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<white>{message}</white>",
    level="INFO",
)

# File output (rotating daily)
logger.add(
    LOG_DIR / "bank_model_factory.log",
    rotation="10 MB",            # rotate after 10 MB
    retention="10 days",         # keep logs for 10 days
    compression="zip",           # compress old logs
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
)

def get_logger():
    """Return the configured Loguru logger."""
    return logger
