import pytest
from datetime import datetime
from src.weixin_summarizer.article import Article

def test_article_initialization():
    """Test Article class initialization."""
    article = Article(
        page_id="test-id",
        title="Test Title"
    )
    assert article.page_id == "test-id"
    assert article.title == "Test Title"
    assert article.processing_status == "pending"

def test_get_text_content(sample_article):
    """Test text content extraction."""
    content = sample_article.get_text_content()
    assert content == "Test content"

def test_to_notion_properties(sample_article):
    """Test conversion to Notion properties."""
    props = sample_article.to_notion_properties()
    
    assert props["Title"]["title"][0]["text"]["content"] == "Test Article"
    assert props["Summary"]["rich_text"][0]["text"]["content"] == "Test summary"
    assert len(props["Tags"]["multi_select"]) == 2
    assert props["Tags"]["multi_select"][0]["name"] == "AI" 