from loguru import logger
import sys

from .config import settings

# 기존 핸들러 제거
logger.remove()

logger.add(
    sys.stdout,
    level=settings["logging"]["level"] or "INFO",
    colorize=True,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level}</level> | "
        "<cyan>{module}.py: line {line}</cyan> | "
        "{message}"
    ),
    enqueue=True,
)


def get_logger(name):
    return logger.bind(name=name) if name else logger
