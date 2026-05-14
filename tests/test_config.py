import pytest
from unittest.mock import patch, MagicMock
from src.core.config import Config


class TestConfig:
    def test_default_values(self):
        cfg = Config(DASHSCOPE_API_KEY="test", BASE_URL="http://test")
        assert cfg.LLM_MODEL == "qwen3.5-plus-2026-04-20"
        assert cfg.PORT == 8347
        assert cfg.USE_AGENT is True

    def test_custom_values(self):
        cfg = Config(
            DASHSCOPE_API_KEY="key123",
            BASE_URL="http://custom",
            LLM_MODEL="custom-model",
            PORT=9999,
        )
        assert cfg.DASHSCOPE_API_KEY == "key123"
        assert cfg.BASE_URL == "http://custom"
        assert cfg.LLM_MODEL == "custom-model"
        assert cfg.PORT == 9999

    def test_cache_ttl_defaults(self):
        cfg = Config(DASHSCOPE_API_KEY="test", BASE_URL="http://test")
        assert cfg.CACHE_TTL_SHORT == 3600 * 24 * 7
        assert cfg.CACHE_TTL_LONG == 3600 * 24 * 30
