import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.weixin_summarizer.notion_client import NotionManager
from src.weixin_summarizer.article import Article
from datetime import datetime

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
async def test_create_article():
    # Setup
    notion_manager = NotionManager()
    
    # Create test article
    article = Article(
        page_id="test-page-id",
        title="Test Article",
        original_url="https://example.com",
        published_date=datetime.now().isoformat(),
        content_blocks=[
            {"type": "paragraph", "text": "Block 1"},
            {"type": "paragraph", "text": "Block 2"},
            # ... add more blocks if needed
        ]
    )

    # Mock the Notion API responses
    with patch.object(notion_manager.client.pages, 'create', new_callable=AsyncMock) as mock_create, \
         patch.object(notion_manager.client.blocks.children, 'append', new_callable=AsyncMock) as mock_append:
        
        # Setup mock return value for page creation
        mock_create.return_value = {"id": "new-page-id"}
        
        # Call the function
        await notion_manager.create_article(article)

        # Verify page creation
        mock_create.assert_called_once_with(
            parent={"database_id": notion_manager.dest_db},
            properties=article.to_notion_properties()
        )

        # Verify blocks were appended
        mock_append.assert_called_once_with(
            block_id="new-page-id",
            children=article.content_blocks
        ) 