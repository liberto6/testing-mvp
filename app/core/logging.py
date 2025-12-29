import logging
from logging.config import dictConfig

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%H:%M:%S",
        },
        "metrics": {
            "format": "\n📊 METRICS --------------------------------------------------\n%(message)s\n-----------------------------------------------------------\n",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
        "metrics_console": {
            "class": "logging.StreamHandler",
            "formatter": "metrics",
            "level": "INFO",
        },
    },
    "loggers": {
        "root": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False
        },
        "uvicorn": {
            "handlers": ["console"],
            "level": "INFO", 
            "propagate": False
        },
        "metrics": {
            "handlers": ["metrics_console"],
            "level": "INFO",
            "propagate": False
        }
    },
}

def setup_logging():
    dictConfig(LOGGING_CONFIG)

logger = logging.getLogger("root")
metrics_logger = logging.getLogger("metrics")
