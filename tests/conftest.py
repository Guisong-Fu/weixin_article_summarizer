import pytest
from datetime import datetime
from typing import Dict, List
from src.weixin_summarizer.article import Article
from src.weixin_summarizer.llm_handler import LLMHandler

@pytest.fixture
def sample_notion_block() -> Dict:
    """Sample Notion block for testing."""
    return {
        "type": "paragraph",
        "paragraph": {
            "rich_text": [
                {
                    "type": "text",
                    "text": {"content": "This is a test paragraph."},
                    "plain_text": "This is a test paragraph."
                }
            ]
        }
    }

@pytest.fixture
def sample_article() -> Article:
    """Sample Article instance for testing."""
    return Article(
        page_id="test-page-id",
        title="Test Article",
        original_url="https://example.com",
        published_date=datetime.now(),
        properties={
            "summary": "Test summary",
            "tags": ["AI", "StartUp"]
        },
        content_blocks=[
            {
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {"content": "Test content"},
                            "plain_text": "Test content"
                        }
                    ]
                }
            }
        ]
    )

@pytest.fixture
def mock_llm_responses() -> Dict[str, str]:
    """Sample LLM responses for testing."""
    return {
        "title": "Generated Test Title",
        "summary": "Generated test summary",
        "tags": "#AI\n#StartUp",
        "rewrite": "Rewritten test content"
    } 