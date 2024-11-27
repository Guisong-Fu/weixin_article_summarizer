import asyncio
import logging
from typing import Optional
from src.weixin_summarizer.llm_handler import LLMHandler
from src.weixin_summarizer.notion_client import NotionManager
from src.weixin_summarizer.processing import ArticleProcessor
from src.weixin_summarizer.utils import setup_logging

async def main(limit: Optional[int] = None):
    """Main application entry point."""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Initialize components
        notion_manager = NotionManager()
        llm_handler = LLMHandler()
        processor = ArticleProcessor(llm_handler)

        # Fetch articles
        logger.info("Fetching articles from Notion...")
        articles = await notion_manager.fetch_articles(limit=limit)
        logger.info(f"Found {len(articles)} articles to process")

        # Process each article
        for article in articles:
            try:
                logger.info(f"Processing article: {article.title}")
                # print(json.dumps(article.content_blocks, indent=4, ensure_ascii=False))
                processed_article = await processor.process_article(article)

                logger.info(f"Updating article in Notion: {processed_article.title}")
                await notion_manager.create_article(processed_article)

                logger.info(f"Archiving original article: {article.title}")
                await notion_manager.archive_article(article)

            except Exception as e:
                logger.error(f"Error processing article {article.title}: {str(e)}")
                continue

        logger.info("Processing completed")

    except Exception as e:
        logger.error(f"Application error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())