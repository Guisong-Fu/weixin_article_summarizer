from typing import List, Dict, Any

from .article import Article
from .llm_handler import LLMHandler
from .prompts import (
    TITLE_PROMPT,
    SUMMARY_PROMPT,
    REWRITE_PROMPT,
    TAGGING_PROMPT
)


class ArticleProcessor:
    """Processes articles using LLM for enhancement."""
    
    def __init__(self, llm_handler: LLMHandler):
        self.llm = llm_handler
        
    async def process_article(self, article: Article) -> Article:
        """Processes an article to generate enhancements."""
        try:
            article.processing_status = "processing"

            content = article.get_text_content()


            # todo: how does this work? Can we use one big prompt and generate all at once?
            # Generate title
            article.title = await self.llm.generate_completion(
                TITLE_PROMPT,
                {"content": content}
            )
            
            # Generate summary
            article.properties["summary"] = await self.llm.generate_completion(
                SUMMARY_PROMPT,
                {"content": content}
            )
            
            # Generate tags
            tags_response = await self.llm.generate_completion(
                TAGGING_PROMPT,
                {"content": content}
            )

            article.properties["tags"] = self._parse_tags(tags_response)
            
            # Rewrite content
            article.content_blocks = await self._rewrite_content(article.content_blocks)
            
            article.processing_status = "completed"
            
        except Exception as e:
            article.processing_status = "failed"
            raise e
            
        return article

    # todo: this one defniitely needs some help
    async def _rewrite_content(
        self, 
        blocks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rewrites the content blocks while preserving structure."""
        rewritten_blocks = []
        
        for block in blocks:
            if block["type"] == "paragraph":
                text = "".join(
                    t["plain_text"] 
                    for t in block["paragraph"]["rich_text"]
                )
                if text.strip():
                    # todo: one question. It rewrites each block independently. Should we append? so it knows the context when processing next block.
                    rewritten_text = await self.llm.generate_completion(
                        REWRITE_PROMPT,
                        {"paragraph": text}
                    )
                    # todo: make sure the Notion structure.
                    rewritten_blocks.append({
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{
                                "type": "text",
                                "text": {"content": rewritten_text}
                            }]
                        }
                    })
            else:
                # Preserve non-paragraph blocks (e.g., images)
                rewritten_blocks.append(block)
                
        return rewritten_blocks
    
    def _parse_tags(self, tags_response: str) -> List[str]:
        """Parses tags from LLM response."""
        valid_tags = {"AI", "StartUp", "Society", "Fun"}
        tags = [
            tag.strip("#").strip() 
            for tag in tags_response.split("\n") 
            if tag.strip().startswith("#")
        ]
        return [tag for tag in tags if tag in valid_tags] 