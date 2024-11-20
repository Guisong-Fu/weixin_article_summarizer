import pytest
import asyncio
import logging
import time
from pathlib import Path
from src.weixin_summarizer.utils import (
    setup_logging,
    rate_limit,
    chunk_list,
    truncate_text,
    safe_get_env,
    validate_notion_block,
    sanitize_text,
    create_error_message,
    measure_time
)


def test_setup_logging(tmp_path):
    """Test logging setup with file output."""
    log_file = tmp_path / "test.log"
    setup_logging(log_file=str(log_file))
    
    logging.info("Test message")
    
    assert log_file.exists()
    assert "Test message" in log_file.read_text()

def test_chunk_list():
    """Test list chunking functionality."""
    test_list = [1, 2, 3, 4, 5]
    chunks = chunk_list(test_list, 2)
    
    assert chunks == [[1, 2], [3, 4], [5]]
    assert chunk_list([], 3) == []

def test_truncate_text():
    """Test text truncation."""
    text = "This is a long text"
    truncated = truncate_text(text, 10)
    
    assert len(truncated) <= 10
    assert truncated.endswith("...")
    assert truncate_text("Short", 10) == "Short"

@pytest.mark.asyncio
async def test_rate_limit():
    """Test rate limiting decorator."""
    calls = []
    
    @rate_limit(calls=2, period=1.0)
    async def test_func():
        calls.append(time.perf_counter())
    
    await asyncio.gather(test_func(), test_func(), test_func())
    
    # Check that calls are properly spaced
    assert len(calls) == 3
    assert calls[2] - calls[0] >= 1.0

def test_safe_get_env():
    """Test environment variable handling."""
    with pytest.raises(ValueError):
        safe_get_env("NONEXISTENT_VAR", required=True)
    
    assert safe_get_env("NONEXISTENT_VAR", default="default") == "default"

def test_validate_notion_block():
    """Test Notion block validation."""
    valid_block = {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"text": "content"}
    }
    invalid_block = {
        "type": "unknown_type"
    }
    
    assert validate_notion_block(valid_block)
    assert not validate_notion_block(invalid_block)

def test_sanitize_text():
    """Test text sanitization."""
    dirty_text = "This  has\tmultiple    spaces\nand\tcontrol\x00characters"
    clean_text = sanitize_text(dirty_text)
    
    assert "  " not in clean_text
    assert "\x00" not in clean_text
    assert clean_text == "This has multiple spaces and control characters"

def test_create_error_message():
    """Test error message formatting."""
    error = ValueError("test error")
    message = create_error_message(error, "testing")
    
    assert "ValueError" in message
    assert "test error" in message
    assert "during testing" in message

@pytest.mark.asyncio
async def test_measure_time(caplog):
    """Test execution time measurement."""
    @measure_time
    async def slow_function():
        await asyncio.sleep(0.1)
        return "done"
    
    with caplog.at_level(logging.DEBUG):
        result = await slow_function()
        
    assert result == "done"
    assert "slow_function took" in caplog.text
    assert "seconds" in caplog.text 