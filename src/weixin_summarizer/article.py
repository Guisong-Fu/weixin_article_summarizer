from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

@dataclass
class Article:
    """Represents an article from Notion with its metadata and content."""
    
    page_id: str
    title: str
    original_url: Optional[str] = None
    published_date: Optional[datetime] = None
    properties: Dict = field(default_factory=dict)
    content_blocks: List[Dict] = field(default_factory=list)
    processing_status: str = "pending"
    
    def get_text_content(self) -> str:
        """Extracts all text content from the article's blocks."""
        text_content = []
        for block in self.content_blocks:
            if block["type"] == "paragraph":
                text = "".join(
                    t["plain_text"] 
                    for t in block["paragraph"]["rich_text"]
                )
                if text.strip():
                    text_content.append(text)
        return "\n\n".join(text_content)

    def to_notion_properties(self) -> Dict:
        """Converts article data to Notion properties format."""
        return {
            "Title": {"title": [{"text": {"content": self.title}}]},
            "Summary": {
                "rich_text": [{"text": {"content": self.properties.get("summary", "")}}]
            },
            "Tags": {
                "multi_select": [{"name": tag} for tag in self.properties.get("tags", [])]
            },
            "Processing Status": {"select": {"name": self.processing_status}},
            "Original URL": {"url": self.original_url} if self.original_url else None,
            "Published Date": {
                "date": {"start": self.published_date.isoformat()}
            } if self.published_date else None,
        } 