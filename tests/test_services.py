import pytest
from unittest.mock import patch, MagicMock
from src.services.auth_service import register_user, login_user
from src.services.session_service import create_session, rename_session


class TestAuthService:
    @patch("src.services.auth_service.get_cache")
    def test_register_empty_username(self, mock_cache):
        with pytest.raises(ValueError, match="用户名和密码不能为空"):
            register_user("", "password")

    @patch("src.services.auth_service.get_cache")
    def test_register_short_username(self, mock_cache):
        with pytest.raises(ValueError, match="用户名长度"):
            register_user("a", "password")

    @patch("src.services.auth_service.get_cache")
    def test_register_short_password(self, mock_cache):
        with pytest.raises(ValueError, match="密码长度"):
            register_user("testuser", "ab")

    @patch("src.services.auth_service.get_cache")
    def test_register_existing_user(self, mock_cache):
        mock_redis = MagicMock()
        mock_redis.exists.return_value = True
        mock_cache.return_value = mock_redis
        with pytest.raises(ValueError, match="用户名已存在"):
            register_user("existing", "password")

    @patch("src.core.security.get_cache")
    @patch("src.services.auth_service.get_cache")
    def test_register_success(self, mock_svc_cache, mock_sec_cache):
        mock_redis = MagicMock()
        mock_redis.exists.return_value = False
        mock_svc_cache.return_value = mock_redis
        mock_sec_cache.return_value = mock_redis
        result = register_user("newuser", "password123")
        assert result["status"] == "ok"
        assert "token" in result

    @patch("src.services.auth_service.get_cache")
    def test_login_nonexistent_user(self, mock_cache):
        mock_redis = MagicMock()
        mock_redis.exists.return_value = False
        mock_cache.return_value = mock_redis
        with pytest.raises(ValueError, match="用户名或密码错误"):
            login_user("nobody", "password")

    @patch("src.core.security.get_cache")
    @patch("src.services.auth_service.get_cache")
    def test_login_wrong_password(self, mock_svc_cache, mock_sec_cache):
        mock_redis = MagicMock()
        mock_redis.exists.return_value = True
        mock_redis.hget.side_effect = lambda k, f: "somesalt" if f == "salt" else "wronghash"
        mock_svc_cache.return_value = mock_redis
        mock_sec_cache.return_value = mock_redis
        with pytest.raises(ValueError, match="用户名或密码错误"):
            login_user("testuser", "wrongpassword")


class TestSessionService:
    @patch("src.services.session_service.get_cache")
    @patch("src.services.session_service.get_settings")
    def test_create_session(self, mock_settings, mock_cache):
        mock_settings.return_value = MagicMock(CACHE_TTL_LONG=86400)
        mock_redis = MagicMock()
        mock_cache.return_value = mock_redis
        result = create_session("testuser")
        assert "session_id" in result

    @patch("src.services.session_service.get_cache")
    def test_rename_session_empty_title(se沙箱的 PATH、环境变量可能和你本地不同lf, mock_cache):
        result = rename_session("sid", "")
        assert result["status"] == "error"

    @patch("src.services.session_service.get_cache")
    def test_rename_session_success(self, mock_cache):
        mock_redis = MagicMock()
        mock_cache.return_value = mock_redis
        result = rename_session("sid", "新标题")
        assert result["status"] == "ok"
