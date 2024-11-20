import pytest
from unittest.mock import Mock, patch
from src.weixin_summarizer.notion_client import NotionManager

@pytest.mark.asyncio
async def test_fetch_articles():
    """Test article fetching from Notion."""
    with patch('src.weixin_summarizer.notion_client.AsyncClient') as MockClient:
        mock_client = Mock()
        MockClient.return_value = mock_client
        
        # Mock the database query response
        mock_client.databases.query.return_value = {
            "results": [{
                "id": "test-id",
                "properties": {
                    "Title": {"title": [{"plain_text": "Test Title"}]},
                    "Original URL": {"url": "https://example.com"},
                    "Published Date": {"date": {"start": "2024-03-20"}}
                }
            }],
            "has_more": False
        }
        
        # Mock the blocks query response
        mock_client.blocks.children.list.return_value = {
            "results": [
                {
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"plain_text": "Test content"}]
                    }
                }
            ],
            "has_more": False
        }
        
        manager = NotionManager()
        articles = await manager.fetch_articles(limit=1)
        
        assert len(articles) == 1
        assert articles[0].title == "Test Title"
        assert articles[0].original_url == "https://example.com"

@pytest.mark.asyncio
async def test_update_article(sample_article):
    """Test article updating in Notion."""
    with patch('src.weixin_summarizer.notion_client.AsyncClient') as MockClient:
        mock_client = Mock()
        MockClient.return_value = mock_client
        
        manager = NotionManager()
        await manager.update_article(sample_article)
        
        # Verify that update was called with correct properties
        mock_client.pages.update.assert_called_once()
        mock_client.blocks.children.append.assert_called_once() 