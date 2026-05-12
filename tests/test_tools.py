import pytest
from unittest.mock import patch, MagicMock
from src.tools.port_check import port_check_logic
from src.tools.server_info import server_system_check_logic
from src.tools.log_analyzer import read_service_log_logic


class TestPortCheck:
    @patch("src.tools.port_check.subprocess.check_output")
    def test_port_occupied(self, mock_output):
        mock_output.return_value = "tcp 0 0 0.0.0.0:8080 0.0.0.0:* LISTEN 1234/python"
        result = port_check_logic(8080)
        assert "8080" in result
        assert "占用" in result

    @patch("src.tools.port_check.subprocess.check_output", side_effect=Exception("not found"))
    def test_port_not_occupied(self, mock_output):
        result = port_check_logic(9999)
        assert "9999" in result
        assert "未被占用" in result


class TestServerInfo:
    @patch("src.tools.server_info.psutil")
    def test_server_check_success(self, mock_psutil):
        mock_psutil.cpu_percent.return_value = 45.0
        mock_mem = MagicMock()
        mock_mem.percent = 60.0
        mock_mem.total = 16 * 1024 * 1024 * 1024
        mock_psutil.virtual_memory.return_value = mock_mem
        mock_disk = MagicMock()
        mock_disk.percent = 70.0
        mock_disk.total = 500 * 1024 * 1024 * 1024
        mock_psutil.disk_usage.return_value = mock_disk
        mock_psutil.pids.return_value = list(range(100))

        result = server_system_check_logic()
        assert "CPU" in result
        assert "内存" in result
        assert "磁盘" in result


class TestLogAnalyzer:
    @patch("src.tools.log_analyzer.subprocess.check_output")
    def test_read_log_success(self, mock_output):
        mock_output.return_value = "line1\nline2\nline3"
        result = read_service_log_logic("/var/log/test.log", 3)
        assert "日志内容" in result

    @patch("src.tools.log_analyzer.subprocess.check_output", side_effect=Exception("no such file"))
    def test_read_log_failure(self, mock_output):
        result = read_service_log_logic("/nonexistent.log", 10)
        assert "读取日志失败" in result
