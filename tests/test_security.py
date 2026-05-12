import pytest
from unittest.mock import patch, MagicMock
from src.core.security import hash_password, verify_password, create_token, get_current_user


class TestSecurity:
    def test_hash_password_deterministic(self):
        result1 = hash_password("test123", "salt1")
        result2 = hash_password("test123", "salt1")
        assert result1 == result2

    def test_hash_password_different_salt(self):
        result1 = hash_password("test123", "salt1")
        result2 = hash_password("test123", "salt2")
        assert result1 != result2

    def test_hash_password_different_password(self):
        result1 = hash_password("test123", "salt1")
        result2 = hash_password("test456", "salt1")
        assert result1 != result2

    @patch("src.core.security.get_cache")
    def test_get_current_user_no_auth(self, mock_cache):
        result = get_current_user(None)
        assert result == "anonymous"

    @patch("src.core.security.get_cache")
    def test_get_current_user_invalid_format(self, mock_cache):
        result = get_current_user("InvalidToken")
        assert result == "anonymous"

    @patch("src.core.security.get_cache")
    def test_create_and_get_token(self, mock_cache):
        mock_redis = MagicMock()
        mock_redis.get.return_value = "testuser"
        mock_cache.return_value = mock_redis

        token = create_token("testuser")
        assert token is not None
        assert len(token) == 64

        username = get_current_user(f"Bearer {token}")
        assert username == "testuser"
