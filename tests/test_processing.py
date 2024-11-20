import pytest
from unittest.mock import Mock
from src.weixin_summarizer.processing import ArticleProcessor
from src.weixin_summarizer.llm_handler import LLMHandler

@pytest.mark.asyncio
async def test_process_article(sample_article, mock_llm_responses):
    """Test article processing."""
    # Mock LLM handler
    mock_llm = Mock(spec=LLMHandler)
    mock_llm.generate_completion = Mock()
    mock_llm.generate_completion.side_effect = [
        mock_llm_responses["title"],
        mock_llm_responses["summary"],
        mock_llm_responses["tags"],
        mock_llm_responses["rewrite"]
    ]
    
    processor = ArticleProcessor(mock_llm)
    processed_article = await processor.process_article(sample_article)
    
    assert processed_article.title == mock_llm_responses["title"]
    assert processed_article.properties["summary"] == mock_llm_responses["summary"]
    assert processed_article.properties["tags"] == ["AI", "StartUp"]
    assert processed_article.processing_status == "completed"

@pytest.mark.asyncio
async def test_process_article_failure(sample_article):
    """Test article processing failure handling."""
    mock_llm = Mock(spec=LLMHandler)
    mock_llm.generate_completion.side_effect = Exception("Test error")
    
    processor = ArticleProcessor(mock_llm)
    
    with pytest.raises(Exception):
        await processor.process_article(sample_article)
    
    assert sample_article.processing_status == "failed"

@pytest.mark.asyncio
async def test_parse_tags():
    """Test tag parsing."""
    processor = ArticleProcessor(Mock())
    tags_response = "#AI\n#InvalidTag\n#StartUp\n#Fun"
    
    parsed_tags = processor._parse_tags(tags_response)
    
    assert set(parsed_tags) == {"AI", "StartUp", "Fun"}
    assert "InvalidTag" not in parsed_tags 