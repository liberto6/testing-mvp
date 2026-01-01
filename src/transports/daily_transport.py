"""
Daily.co WebRTC Transport for Pipecat
Production-grade WebRTC using Daily.co infrastructure
"""

from typing import Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DailyTransportConfig:
    """Configuration for Daily transport"""
    api_key: str
    room_url: Optional[str] = None
    sample_rate: int = 16000
    channels: int = 1


class DailyTransport:
    """
    Daily.co WebRTC transport wrapper for Pipecat

    Note: Pipecat has built-in Daily transport support
    This is a configuration helper
    """

    def __init__(self, config: DailyTransportConfig):
        self.config = config

    def get_pipecat_transport_config(self) -> dict:
        """Get configuration for Pipecat's built-in Daily transport"""

        try:
            from pipecat.transports.daily_transport import DailyTransportConfig

            # Return config for Pipecat's Daily transport
            return {
                "api_key": self.config.api_key,
                "room_url": self.config.room_url,
                "sample_rate": self.config.sample_rate,
                "channels": self.config.channels
            }

        except ImportError:
            logger.error("❌ Pipecat Daily transport not available")
            logger.error("   Install with: pip install pipecat-ai[daily]")
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n🧪 Daily.co Transport Module\n")
    print("✅ This module provides Daily.co WebRTC transport configuration")
    print("📝 Pipecat has built-in Daily support")
    print("\n💡 Install with: pip install pipecat-ai[daily]")
    print("\nExample usage:")
    print("""
    from pipecat.transports.daily_transport import DailyTransport
    from src.transports.daily_transport import DailyTransportConfig

    config = DailyTransportConfig(
        api_key="your_daily_api_key",
        room_url="https://your-domain.daily.co/room-name"
    )

    transport = DailyTransport(
        api_key=config.api_key,
        room_url=config.room_url
    )
    """)
