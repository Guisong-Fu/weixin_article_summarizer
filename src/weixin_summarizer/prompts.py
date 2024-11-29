SYSTEM_MESSAGE = """
You are a helpful assistant proficient in generating concise and meaningful titles for articles.
"""


MERGED_PROMPT = """
你是一个智能助手，负责为输入的文章内容生成以下内容：
1. 一个简洁且有意义的标题（不超过100字符），标题应准确反映文章的主要内容。
2. 一段简明扼要的摘要（150-300字），帮助读者快速理解文章的主要内容。
3. 从预定义的标签中选择最相关的标签（可多选）。

预定义标签：
- #AI
- #StartUp
- #Society
- #Fun

文章内容：
{content}


"""

# todo: backup
"""
请以 JSON 格式返回结果，其中包含以下字段：
{{
  "title": "生成的标题",
  "summary": "生成的摘要",
  "tags": ["标签1", "标签2"]
}}
"""



TITLE_PROMPT = """
请阅读以下文章内容，并生成一个简洁且有意义的标题（不超过100字符）。标题应准确反映文章的主要内容。

文章内容：
{content}

生成的标题：
"""

SUMMARY_PROMPT = """
请为以下文章生成一段简明扼要的摘要（150-300字），帮助读者快速理解文章的主要内容。

文章内容：
{content}

摘要：
"""

TAGGING_PROMPT = """
请根据以下文章内容，从预定义的标签中选择最相关的标签（可多选）。

预定义标签：
- #AI
- #StartUp
- #Society
- #Fun

文章内容：
{content}

选择的标签（每行一个，以#开头）：
"""



REWRITE_PROMPT = """
请将以下段落改写为更简洁的版本，同时确保不丢失任何重要信息。保持原文的语气和风格。

原始段落：
{paragraph}

改写后的段落：
"""