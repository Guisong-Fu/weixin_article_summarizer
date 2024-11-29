import logging
import os
from typing import List, Dict, Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from notion_client import AsyncClient
from .article import Article
from .utils import chunk_list

from dotenv import load_dotenv
load_dotenv()


class NotionManager:
    """Manages interactions with Notion API."""

    def __init__(self):
        self.client = AsyncClient(auth=os.getenv("NOTION_TOKEN"))
        self.source_db = os.getenv("SOURCE_DATABASE_ID")
        self.dest_db = os.getenv("DESTINATION_DATABASE_ID")
        self.archive_db = os.getenv("ARCHIVE_DATABASE_ID")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )

    # todo: double check this, how blocks work
    async def fetch_articles(self, limit: Optional[int] = None) -> List[Article]:
        """Fetches articles from the source database."""
        articles = []
        query = {"database_id": self.source_db}

        if limit:
            query["page_size"] = min(limit, 100)

        has_more = True
        # todo: what is this `next_cursor` used for?
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

        # todo: we may need more propertities
        return Article(
            page_id=page["id"],
            title=self._extract_title(page),
            original_url=self._extract_url(page),
            published_date=self._extract_date(page),
            content_blocks=blocks
        )

    # todo: double check this! Instead of using Any as type, maybe we can create a Class for this?
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
    async def create_article(self, article: Article) -> None:
        """Creates a new article in the destination database."""
        # Create new page with properties
        response = await self.client.pages.create(
            parent={"database_id": self.dest_db},
            properties=article.to_notion_properties()
        )

        # Get the new page ID
        new_page_id = response["id"]


        # Add content blocks in chunks of 100
        # here is the error we got: APIResponseError: body failed validation: body.children should be an array, instead was `{"object":"block","id":"146520c8-6831-81e6-9541-d38a0a3...`.
        # for chunk in chunk_list(article.content_blocks, 100):
        #     await self.client.blocks.children.append(
        #         block_id=new_page_id,
        #         children=chunk
        #     )

        # for block in article.content_blocks:
        #     await self.client.blocks.children.append(
        #         block_id=new_page_id,
        #         children=block
        #     )

        await self.client.blocks.children.append(
            block_id=new_page_id,
            children=article.content_blocks)


    def _extract_title(self, page: Dict[str, Any]) -> str:
        """Extracts title from page properties."""
        title_prop = page["properties"].get("Name", {}).get("title", [])
        return title_prop[0]["plain_text"] if title_prop else "Untitled"

    def _extract_url(self, page: Dict[str, Any]) -> Optional[str]:
        """Extracts URL from page properties."""
        return page["properties"].get("文章链接", {}).get("url")

    def _extract_date(self, page: Dict[str, Any]) -> Optional[str]:
        """Extracts published date from page properties."""
        return page["properties"].get("创建时间", {}).get("created_time")
    # todo: maybe we need more properties?


    # @retry(
    #     stop=stop_after_attempt(3),
    #     wait=wait_exponential(multiplier=1, min=4, max=10)
    # )
    # todo: probably only save the page_id somewhere. No way to archive
    async def archive_article(self, article: Article) -> None:
        """Moves an article to the archive database."""
        # Retrieve the original page's properties
        logging.error("XXXX", article.page_id)

        original_page = await self.client.pages.retrieve(page_id=article.page_id)
        properties = original_page['properties']

        logging.error("properties", properties)
        # Create a new page in the archive database with the same properties
        await self.client.pages.create(
            parent={"database_id": self.archive_db},
            properties=properties
        )

        # Optionally, delete or archive the original page
        await self.client.pages.update(
            page_id=article.page_id,
            archived=True
        )