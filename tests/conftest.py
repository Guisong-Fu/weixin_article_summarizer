from datetime import datetime
from typing import Dict, List, Any

import pytest

from src.weixin_summarizer.article import Article


# @pytest.fixture
def sample_notion_block() -> List[Dict[str, Any]]:
    return [
    {
        "object": "block",
        "id": "141520c8-6831-818a-a423-f73fcd446d33",
        "parent":
        {
            "type": "page_id",
            "page_id": "141520c8-6831-817d-aa99-c4f1db6460e4"
        },
        "created_time": "2024-11-17T05:12:00.000Z",
        "last_edited_time": "2024-11-17T05:12:00.000Z",
        "created_by":
        {
            "object": "user",
            "id": "140d872b-594c-81bb-a38b-002721962b2f"
        },
        "last_edited_by":
        {
            "object": "user",
            "id": "140d872b-594c-81bb-a38b-002721962b2f"
        },
        "has_children": False,
        "archived": False,
        "in_trash": False,
        "type": "callout",
        "callout":
        {
            "rich_text":
            [
                {
                    "type": "text",
                    "text":
                    {
                        "content": "您未设置自己的图床，本文图片未能帮您转存，可前往 小程序-我的-图床配置 进行配置。",
                        "link": None
                    },
                    "annotations":
                    {
                        "bold": False,
                        "italic": False,
                        "strikethrough": False,
                        "underline": False,
                        "code": False,
                        "color": "default"
                    },
                    "plain_text": "您未设置自己的图床，本文图片未能帮您转存，可前往 小程序-我的-图床配置 进行配置。",
                    "href": None
                }
            ],
            "icon":
            {
                "type": "emoji",
                "emoji": "⚠️"
            },
            "color": "yellow_background"
        }
    },
    {
        "object": "block",
        "id": "141520c8-6831-81ad-b8ec-e918f62b253a",
        "parent":
        {
            "type": "page_id",
            "page_id": "141520c8-6831-817d-aa99-c4f1db6460e4"
        },
        "created_time": "2024-11-17T05:12:00.000Z",
        "last_edited_time": "2024-11-17T05:12:00.000Z",
        "created_by":
        {
            "object": "user",
            "id": "140d872b-594c-81bb-a38b-002721962b2f"
        },
        "last_edited_by":
        {
            "object": "user",
            "id": "140d872b-594c-81bb-a38b-002721962b2f"
        },
        "has_children": False,
        "archived": False,
        "in_trash": False,
        "type": "image",
        "image":
        {
            "caption":
            [],
            "type": "external",
            "external":
            {
                "url": "https://mmbiz.qpic.cn/mmbiz_png/b2YlTLuGbKD1gk1BicNDlq0VdXDTTGWONL4IRozic33X70ZGMfYGRSVEkCicMHhX8BPD10gUJCYZZ8o70mAEibUPpg/640.png?wx_fmt=png&from=appmsg"
            }
        }
    },
    {
        "object": "block",
        "id": "141520c8-6831-81ed-b21f-c61fbbe316b6",
        "parent":
        {
            "type": "page_id",
            "page_id": "141520c8-6831-817d-aa99-c4f1db6460e4"
        },
        "created_time": "2024-11-17T05:12:00.000Z",
        "last_edited_time": "2024-11-17T05:12:00.000Z",
        "created_by":
        {
            "object": "user",
            "id": "140d872b-594c-81bb-a38b-002721962b2f"
        },
        "last_edited_by":
        {
            "object": "user",
            "id": "140d872b-594c-81bb-a38b-002721962b2f"
        },
        "has_children": False,
        "archived": False,
        "in_trash": False,
        "type": "paragraph",
        "paragraph":
        {
            "rich_text":
            [
                {
                    "type": "text",
                    "text":
                    {
                        "content": "本文来自微信公众号：巨潮WAVE （ID：WAVE-BIZ），作者：老鱼儿，编辑：杨旭然，题图来自：AI生成",
                        "link": None
                    },
                    "annotations":
                    {
                        "bold": False,
                        "italic": False,
                        "strikethrough": False,
                        "underline": False,
                        "code": False,
                        "color": "default"
                    },
                    "plain_text": "本文来自微信公众号：巨潮WAVE （ID：WAVE-BIZ），作者：老鱼儿，编辑：杨旭然，题图来自：AI生成",
                    "href": None
                }
            ],
            "color": "default"
        }
    },
    {
        "object": "block",
        "id": "141520c8-6831-81d3-9d34-c31982db44b6",
        "parent":
        {
            "type": "page_id",
            "page_id": "141520c8-6831-817d-aa99-c4f1db6460e4"
        },
        "created_time": "2024-11-17T05:12:00.000Z",
        "last_edited_time": "2024-11-17T05:12:00.000Z",
        "created_by":
        {
            "object": "user",
            "id": "140d872b-594c-81bb-a38b-002721962b2f"
        },
        "last_edited_by":
        {
            "object": "user",
            "id": "140d872b-594c-81bb-a38b-002721962b2f"
        },
        "has_children": False,
        "archived": False,
        "in_trash": False,
        "type": "paragraph",
        "paragraph":
        {
            "rich_text":
            [
                {
                    "type": "text",
                    "text":
                    {
                        "content": "英国作为世界上最重要的留学目的国之一，其国内的一些大学可谓是世界各国学生的“朝圣之地”。在中国，人们对英国的教育和留学项目尤其关注。",
                        "link": None
                    },
                    "annotations":
                    {
                        "bold": False,
                        "italic": False,
                        "strikethrough": False,
                        "underline": False,
                        "code": False,
                        "color": "default"
                    },
                    "plain_text": "英国作为世界上最重要的留学目的国之一，其国内的一些大学可谓是世界各国学生的“朝圣之地”。在中国，人们对英国的教育和留学项目尤其关注。",
                    "href": None
                }
            ],
            "color": "default"
        }
    }
]

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