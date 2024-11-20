import os
from typing import List, Dict, Any, Optional
from notion_client import AsyncClient
from tenacity import retry, stop_after_attempt, wait_exponential
from .article import Article
from .utils import chunk_list

class NotionManager:
    """Manages interactions with Notion API."""
    
    def __init__(self):
        self.client = AsyncClient(auth=os.getenv("NOTION_TOKEN"))
        self.source_db = os.getenv("SOURCE_DATABASE_ID")
        self.dest_db = os.getenv("DESTINATION_DATABASE_ID")
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def fetch_articles(self, limit: Optional[int] = None) -> List[Article]:
        """Fetches articles from the source database."""
        articles = []
        query = {"database_id": self.source_db}
        
        if limit:
            query["page_size"] = min(limit, 100)
            
        has_more = True
        next_cursor = None
        
        while has_more and (not limit or len(articles) < limit):
            if next_cursor:
                query["start_cursor"] = next_cursor
                
            response = await self.client.databases.query(**query)
            
            for page in response["results"]:
                article = await self._page_to_article(page)
                articles.append(article)
                
            has_more = response["has_more"]
            next_cursor = response["next_cursor"]
            
        return articles[:limit] if limit else articles
    
    async def _page_to_article(self, page: Dict[str, Any]) -> Article:
        """Converts a Notion page to an Article object."""
        blocks = await self._fetch_page_blocks(page["id"])
        
        return Article(
            page_id=page["id"],
            title=self._extract_title(page),
            original_url=self._extract_url(page),
            published_date=self._extract_date(page),
            content_blocks=blocks
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _fetch_page_blocks(self, page_id: str) -> List[Dict[str, Any]]:
        """Fetches all blocks from a Notion page."""
        blocks = []
        has_more = True
        next_cursor = None
        
        while has_more:
            response = await self.client.blocks.children.list(
                block_id=page_id,
                start_cursor=next_cursor
            )
            
            blocks.extend(response["results"])
            has_more = response["has_more"]
            next_cursor = response["next_cursor"]
            
        return blocks
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def update_article(self, article: Article) -> None:
        """Updates an article in the destination database."""
        # Update properties
        await self.client.pages.update(
            page_id=article.page_id,
            properties=article.to_notion_properties()
        )
        
        # Update content blocks in chunks
        existing_blocks = await self._fetch_page_blocks(article.page_id)
        
        # Delete existing blocks
        for block in existing_blocks:
            await self.client.blocks.delete(block_id=block["id"])
            
        # Add new blocks in chunks of 100
        for chunk in chunk_list(article.content_blocks, 100):
            await self.client.blocks.children.append(
                block_id=article.page_id,
                children=chunk
            )
            
    def _extract_title(self, page: Dict[str, Any]) -> str:
        """Extracts title from page properties."""
        title_prop = page["properties"].get("Title", {}).get("title", [])
        return title_prop[0]["plain_text"] if title_prop else "Untitled"
    
    def _extract_url(self, page: Dict[str, Any]) -> Optional[str]:
        """Extracts URL from page properties."""
        return page["properties"].get("Original URL", {}).get("url")
    
    def _extract_date(self, page: Dict[str, Any]) -> Optional[str]:
        """Extracts published date from page properties."""
        date_prop = page["properties"].get("Published Date", {}).get("date")
        return date_prop["start"] if date_prop else None 