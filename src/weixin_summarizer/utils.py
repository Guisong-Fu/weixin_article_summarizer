import logging
import time
import os
from typing import List, Any, Callable, TypeVar, Optional
from functools import wraps
from pathlib import Path
import asyncio

T = TypeVar('T')

def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> None:
    """
    Configures logging for the application.
    
    Args:
        log_level: The logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Optional path to log file. If None, logs to console
        log_format: Optional custom log format string
    """
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    format_str = log_format or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(format_str))
    handlers.append(console_handler)
    
    # File handler if log_file specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_str))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        format=format_str
    )

def rate_limit(calls: int, period: float = 1.0) -> Callable:
    """
    Decorator to limit the rate of function calls.
    
    Args:
        calls: Maximum number of calls allowed in the period
        period: Time period in seconds
        
    Returns:
        Decorated function with rate limiting
    """
    min_interval = period / float(calls)
    
    def decorator(func: Callable) -> Callable:
        last_time_called = [0.0]
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            elapsed = time.perf_counter() - last_time_called[0]
            left_to_wait = min_interval - elapsed
            
            if left_to_wait > 0:
                await asyncio.sleep(left_to_wait)
                
            result = await func(*args, **kwargs)
            last_time_called[0] = time.perf_counter()
            return result
            
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            elapsed = time.perf_counter() - last_time_called[0]
            left_to_wait = min_interval - elapsed
            
            if left_to_wait > 0:
                time.sleep(left_to_wait)
                
            result = func(*args, **kwargs)
            last_time_called[0] = time.perf_counter()
            return result
            
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

def chunk_list(lst: List[T], chunk_size: int) -> List[List[T]]:
    """
    Splits a list into smaller chunks of specified size.
    
    Args:
        lst: List to split
        chunk_size: Maximum size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncates text to specified length, adding suffix if truncated.
    
    Args:
        text: Text to truncate
        max_length: Maximum length of resulting text
        suffix: String to append if text is truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def safe_get_env(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    Safely retrieves environment variable with error handling.
    
    Args:
        key: Environment variable key
        default: Default value if key not found
        required: Whether the environment variable is required
        
    Returns:
        Environment variable value or default
        
    Raises:
        ValueError: If required is True and environment variable is not set
    """
    value = os.getenv(key, default)
    if required and value is None:
        raise ValueError(f"Required environment variable '{key}' is not set")
    return value

def validate_notion_block(block: dict) -> bool:
    """
    Validates the structure of a Notion block.
    
    Args:
        block: Notion block dictionary
        
    Returns:
        True if block is valid, False otherwise
    """
    required_keys = {"type", "object"}
    if not all(key in block for key in required_keys):
        return False
        
    if block["type"] not in {"paragraph", "image", "heading_1", "heading_2", "heading_3"}:
        return False
        
    return True

def sanitize_text(text: str) -> str:
    """
    Sanitizes text by removing unwanted characters and normalizing whitespace.
    
    Args:
        text: Text to sanitize
        
    Returns:
        Sanitized text
    """
    # Remove control characters
    text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")
    
    # Normalize whitespace
    text = " ".join(text.split())
    
    return text.strip()

def create_error_message(error: Exception, context: Optional[str] = None) -> str:
    """
    Creates a formatted error message with context.
    
    Args:
        error: Exception object
        context: Optional context information
        
    Returns:
        Formatted error message
    """
    error_type = type(error).__name__
    error_message = str(error)
    context_info = f" during {context}" if context else ""
    
    return f"{error_type}{context_info}: {error_message}"

def measure_time(func: Callable) -> Callable:
    """
    Decorator to measure execution time of a function.
    
    Args:
        func: Function to measure
        
    Returns:
        Decorated function that logs execution time
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        logging.debug(f"{func.__name__} took {elapsed_time:.2f} seconds")
        return result
        
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        logging.debug(f"{func.__name__} took {elapsed_time:.2f} seconds")
        return result
        
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper 