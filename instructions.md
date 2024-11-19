# 1. Project Overview

You will act as a senior python developer with extensive experience in LLM and LangChain, and you also familar with Notion API. You will be building a system that can process articles stored in one Notion database, and enhance them by generating new titles, summaries, rewritten content, and tags, and write them to another Notion database.


## 1.1. Introduction
The Automated Article Processing and Enhancement System is designed to process articles extracted from WeChat Public Accounts, stored in a Notion database. The system aims to enhance the articles by generating meaningful titles, summaries, rewriting content for brevity without losing information(keep the original text and image order), and tagging articles based on predefined categories. The processing leverages Large Language Models (LLMs) through the LangChain framework, integrating both local and cloud-based models. Local LLMs will be hosted through Ollama.

## 1.2. Background
A pipeline is already in place that extracts content (text and images) from blogs (WeChat Public Accounts) and stores them in a Notion database. The current challenge is that the original articles may have non-descriptive titles, and users need a quick way to understand the content without reading the entire article.

# 2. Objectives
* Generate Meaningful Titles: Create concise and informative titles that accurately reflect the content of each article.
* Summarize Articles: Provide one-paragraph summaries to help users quickly assess their interest in the full article.
* Rewrite Articles: Rewrite each paragraph to be shorter while retaining all original information.
* Tag Articles: Automatically tag articles using a predefined set of tags for easier categorization and retrieval.
* Maintain Content Integrity: Preserve the order of texts and images, including image references, to verify content authenticity.
* Flexible LLM Integration: Easily switch between different LLMs (OpenAI, Ollama with Qwen, etc.) for processing.
* Local Deployment: Run the system locally on a MacBook Pro with M3 Max chip and 128GB memory.
* Manual Execution: Allow for manual triggering of the processing pipeline, with future potential for automation.

# 3. Scope

## 3.1. In Scope
* Processing articles currently stored in the Notion database.
* Generating new titles, summaries, rewritten content, and tags.
* Integration with Notion API for reading and updating content.
* Using LangChain framework for LLM interactions.
* Implementing predefined tags (AI, StartUp, Other).
* Supporting content in Chinese, with potential for future multilingual support.
* Using LangSmith for observability and logging.

## 3.2. Out of Scope
* Processing images (e.g., OCR, image recognition).
* Implementing user interfaces beyond Notion.
* Handling scaling issues or high-volume data processing.
* Automated scheduling or event-driven execution (to be considered in future versions).
* Advanced error handling for rare edge cases.
* Compliance with regulations beyond standard data privacy practices.

# 4. Functional Requirements

## 4.1. Article Retrieval
* FR1.1: The system shall retrieve articles from the existing Notion database without altering the database structure.

## 4.2. Title Generation
* FR2.1: Generate a concise and meaningful title for each article that accurately reflects its content.
* FR2.2: Write the generated title to a designated field in the new Notion database.

## 4.3. Article Summarization
* FR3.1: Produce a one-paragraph summary of each article.
* FR3.2: Write the summary to a designated field in the new Notion database.

## 4.4. Article Rewriting
* FR4.1: Rewrite each paragraph of the article to be shorter while retaining all original information.
* FR4.2: Ensure that the order of text and images remains unchanged.
* FR4.3: Preserve image references for content verification.
* FR4.4: Write the rewritten content to a designated field in the new Notion database.

## 4.5. Article Tagging
* FR5.1: Automatically generate tags using the predefined tags based on content analysis.
* FR5.2: Write the tags to a designated field in the new Notion database.

## 4.6. LLM Integration
* FR6.1: Integrate with OpenAI's GPT-3.5-Turbo model via LangChain.
* FR6.2: Allow easy switching between LLMs for testing purposes.

## 4.7. Execution Control
* FR7.1: Enable manual triggering of the processing pipeline, `app.py` is the entry point of the application.

## 4.8. Logging and Monitoring
* FR8.1: Implement logging using LangSmith for observability.
* FR8.2: Log key events, errors, and LLM interactions.

# 5. System Architecture

## 5.1. Overview
The system comprises the following components:

* Data Retrieval Module: Fetches articles from the Notion database.
* Processing Pipeline: Handles title generation, summarization, rewriting, and tagging using LLMs.
* Data Update Module: Updates the Notion database with processed data.
* LLM Handler: Manages interactions with different LLMs via LangChain.
* Logging and Monitoring Module: Uses LangSmith for observability and logging.

## 5.2. Architecture Diagram
+---------------------+
|                     |
|    Manual Trigger   |
|  (User runs app.py)|
|                     |
+----------+----------+
           |
           v
+----------+----------+
|                     |
| Configuration &     |
| Environment Setup   |
| (Load .env file)    |
|                     |
+----------+----------+
           |
           v
+----------+----------+
|                     |
|   Notion API Client |
|                     |
+----+---------+------+
     |         |
     |         |
     v         v
+----+----+ +--+-----+
|         | |        |
|  Data   | |  Data  |
| Retrieval| | Update|
|  Module | | Module |
|         | |        |
+----+----+ +----+---+
     |           |
     |           |
     v           v
+----+-----------+---+
|                    |
|   Article Data     |
|       Model        |
|    (Article obj)   |
|                    |
+----+-----------+---+
     |            |
     |            |
     v            v
+----+-----------+---+
|                    |
|   Processing Module|
| (Title Gen, Summ,  |
|  Rewrite, Tagging) |
|                    |
+----+-----------+---+
     |            |
     |            |
     v            v
+----+-----------+---+
|                    |
|     LLM Handler    |
|  (OpenAI, Ollama)  |
|                    |
+----+-----------+---+
     |            |
     |            |
     v            v
+----+-----------+---+
|                    |
|    Prompts Module  |
|  (Prompt Templates)|
|                    |
+----+-----------+---+
     |
     v
+----+-----------+---+
|                    |
| Logging & Monitoring|
|    (LangSmith)     |
|                    |
+--------------------+


# this should be detailed out -> 

# 6. Python Packages & Dependencies
Python Packages:
* Poetry
* langchain
* openai
* notion-client
* python-dotenv

Please feel free to suggest more packages if needed.

# 7. Implementation Plan




## 7.1. Code Structure
* app.py: Entry point of the application.
* article.py: Contains the Article data model.
* notion_client.py: Handles interactions with the Notion API.
* processing.py: Contains functions for title generation, summarization, rewriting, and tagging.
* llm_handler.py: Manages LLM initialization and switching.
* prompts.py: Stores prompt templates.
* utils.py: Utility functions.

# 8. Testing Plan

## 8.1. Unit Testing
* Write unit tests for each function in processing.py, notion_client.py, and llm_handler.py.
* Use a testing framework like pytest.

## 8.2. Integration Testing
* Test the end-to-end processing of a single article.
* Verify that the article in Notion is updated correctly.




# 11. Appendices

## Appendix A: Sample Code

## A.1. main.py
```python
import os
from dotenv import load_dotenv
from notion_client import Client
from langchain import LLMChain
from langchain.llms import OpenAI
# from langchain.llms import Ollama  # Uncomment if Ollama integration is available
from langchain.callbacks import LangSmithCallbackHandler
from article import Article
from notion_client_module import fetch_articles, update_page_properties, update_page_content
from processing import process_article
from llm_handler import initialize_llm

def main():
    # Load environment variables
    load_dotenv()

    # Initialize Notion client
    notion = Client(auth=os.getenv('NOTION_TOKEN'))

    # Initialize LLM
    llm = initialize_llm()

    # Fetch articles
    database_id = os.getenv('DATABASE_ID')
    articles = fetch_articles(notion, database_id)

    # Process articles
    for article in articles:
        try:
            processed_article = process_article(article, llm)
            update_page_properties(notion, processed_article)
            update_page_content(notion, processed_article)
            print(f"Processed article {article.page_id} successfully.")
        except Exception as e:
            print(f"Error processing article {article.page_id}: {e}")

if __name__ == '__main__':
    main()
```

## A.2. processing.py
```python
from langchain import LLMChain
from prompts import title_prompt, summary_prompt, rewrite_prompt, tagging_prompt

def generate_title(llm, content):
    chain = LLMChain(llm=llm, prompt=title_prompt)
    return chain.run(content).strip()

def generate_summary(llm, content):
    chain = LLMChain(llm=llm, prompt=summary_prompt)
    return chain.run(content).strip()

def rewrite_paragraph(llm, paragraph):
    chain = LLMChain(llm=llm, prompt=rewrite_prompt)
    return chain.run(paragraph).strip()

def generate_tags(llm, content):
    chain = LLMChain(llm=llm, prompt=tagging_prompt)
    tags_response = chain.run(content)
    tags = [tag.strip() for tag in tags_response.split('\n') if tag.startswith('#')]
    return tags

def process_article(article, llm):
    # Combine text content
    content = ''
    for block in article.content_blocks:
        if block['type'] == 'paragraph':
            texts = block['paragraph']['text']
            content += ''.join([text['plain_text'] for text in texts]) + '\n'

    # Generate new title
    new_title = generate_title(llm, content)

    # Generate summary
    summary = generate_summary(llm, content)

    # Generate tags
    tags = generate_tags(llm, content)

    # Rewrite content blocks
    rewritten_blocks = []
    for block in article.content_blocks:
        if block['type'] == 'paragraph':
            texts = block['paragraph']['text']
            paragraph_text = ''.join([text['plain_text'] for text in texts])
            rewritten_paragraph = rewrite_paragraph(llm, paragraph_text)
            new_block = {
                'object': 'block',
                'type': 'paragraph',
                'paragraph': {
                    'text': [{
                        'type': 'text',
                        'text': {
                            'content': rewritten_paragraph
                        }
                    }]
                }
            }
            rewritten_blocks.append(new_block)
        else:
            # Preserve non-paragraph blocks
            rewritten_blocks.append(block)

    # Update the article object
    article.title = new_title
    article.properties['Summary'] = summary
    article.properties['Tags'] = tags
    article.content_blocks = rewritten_blocks

    return article
```

## A.3. prompts.py
```python
from langchain.prompts import PromptTemplate

# Title Generation Prompt
title_prompt = PromptTemplate(
    input_variables=["content"],
    template="""
请阅读以下文章内容，并为其生成一个简洁且有意义的标题，使其准确反映文章的主要内容。

文章内容：
{content}

生成的标题：
"""
)

# Summary Generation Prompt
summary_prompt = PromptTemplate(
    input_variables=["content"],
    template="""
请为以下文章内容生成一段话的摘要，帮助读者快速了解文章的主要内容。

文章内容：
{content}

摘要：
"""
)

# Paragraph Rewriting Prompt
rewrite_prompt = PromptTemplate(
    input_variables=["paragraph"],
    template="""
请将以下段落改写为更简短的版本，同时确保不丢失任何信息。

原始段落：
{paragraph}

改写后的段落：
"""
)

# Tagging Prompt
tagging_prompt = PromptTemplate(
    input_variables=["content"],
    template="""
请根据以下文章内容，从预定义的标签列表中选择最合适的标签。预定义标签列表如下：

- #AI
- #StartUp
- #Society
- #Fun

文章内容：
{content}

选择的标签（仅选择最相关的标签）：
"""
)
```

## Appendix B: Environment Setup Instructions

1. Install Python 3.8+
```bash
python3 --version
```

2. Install Poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Clone the Repository
```bash
git clone https://github.com/your-repo/article-