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


# 6. Python Packages & Dependencies
Python Packages:
* Poetry
* langchain
* openai
* notion-client
* python-dotenv

Please feel free to suggest more packages if needed.

# 7. Implementation Plan

## 7.1.1 Code Structure
* app.py: Entry point of the application.
* article.py: Contains the Article data model.
* notion_client.py: Handles interactions with the Notion API.
* processing.py: Contains functions for title generation, summarization, rewriting, and tagging.
* llm_handler.py: Manages LLM initialization and switching.
* prompts.py: Stores prompt templates.
* utils.py: Utility functions.

## 7.1.2 Project Structure

project/
├── app.py
├── llm_handler.py
├── utils.py
├── processing.py
├── notion_client.py
├── article.py
├── prompts.py
├── .env
└── pyproject.toml

## 7.2. Implementation Details

### 7.2.1. Interact with Notion via Notion API, `notion_client.py`
- Connect to Notion database(`SOURCE_DATABASE_ID` from `.env`), search for articles(pages) available in that database
- Write back to another Notion database(`DESTINATION_DATABASE_ID` from `.env`)
- here is some sample code, and you can also take a look at  `inspect_notion_database.py`

```python
# notion_client.py

from notion_client import Client
from article import Article
from typing import List
import time

def get_page_blocks(notion: Client, page_id: str) -> List[dict]:
    """
    Retrieves all blocks (content elements) from a Notion page.

    Args:
        notion (Client): The Notion client instance.
        page_id (str): The ID of the Notion page.

    Returns:
        List[dict]: A list of block objects from the page.
    """
    blocks = []
    has_more = True
    next_cursor = None

    while has_more:
        response = notion.blocks.children.list(
            block_id=page_id,
            start_cursor=next_cursor
        )
        blocks.extend(response['results'])
        has_more = response.get('has_more', False)
        next_cursor = response.get('next_cursor', None)

        # Notion API rate limit handling
        time.sleep(0.2)

    return blocks

def fetch_articles(notion: Client, database_id: str) -> List[Article]:
    """
    Fetches articles from a Notion database and converts them into Article objects.

    Args:
        notion (Client): The Notion client instance.
        database_id (str): The ID of the Notion database.

    Returns:
        List[Article]: A list of Article objects.
    """
    articles = []
    has_more = True
    next_cursor = None

    while has_more:
        response = notion.databases.query(
            database_id=database_id,
            start_cursor=next_cursor
        )
        pages = response['results']
        for page in pages:
            page_id = page['id']
            title_property = page['properties'].get('Name', {}).get('title', [])
            title = title_property[0]['plain_text'] if title_property else 'Untitled'
            properties = page['properties']
            content_blocks = get_page_blocks(notion, page_id)
            article = Article(
                page_id=page_id,
                title=title,
                properties=properties,
                content_blocks=content_blocks
            )
            articles.append(article)
        has_more = response.get('has_more', False)
        next_cursor = response.get('next_cursor', None)

        # Notion API rate limit handling
        time.sleep(0.2)

    return articles

def update_page_properties(notion: Client, article: Article):
    """
    Updates the properties of a Notion page (article).

    Args:
        notion (Client): The Notion client instance.
        article (Article): The Article object with updated properties.
    """
    properties = {
        'Name': {
            'title': [
                {
                    'text': {
                        'content': article.title
                    }
                }
            ]
        },
        'Summary': {
            'rich_text': [
                {
                    'text': {
                        'content': article.properties.get('Summary', '')
                    }
                }
            ]
        },
        'Tags': {
            'multi_select': [{'name': tag} for tag in article.properties.get('Tags', [])]
        }
    }
    try:
        notion.pages.update(page_id=article.page_id, properties=properties)
    except Exception as e:
        print(f"Error updating properties for page {article.page_id}: {e}")

def update_page_content(notion: Client, article: Article):
    """
    Updates the content blocks of a Notion page while preserving the order of texts and images.

    Args:
        notion (Client): The Notion client instance.
        article (Article): The Article object with updated content_blocks.
    """
    # Fetch existing blocks
    existing_blocks = get_page_blocks(notion, article.page_id)

    # Delete existing blocks
    for block in existing_blocks:
        try:
            notion.blocks.delete(block_id=block['id'])
            time.sleep(0.2)  # Notion API rate limit handling
        except Exception as e:
            print(f"Error deleting block {block['id']}: {e}")

    # Add new blocks
    # Notion API allows adding up to 100 blocks at a time
    blocks_to_add = article.content_blocks
    for i in range(0, len(blocks_to_add), 100):
        chunk = blocks_to_add[i:i+100]
        try:
            notion.blocks.children.append(
                block_id=article.page_id,
                children=chunk
            )
            time.sleep(0.2)  # Notion API rate limit handling
        except Exception as e:
            print(f"Error adding blocks to page {article.page_id}: {e}")
```

### 7.2.2. Manage LLM related configuration, `llm_handler.py`
- initialize configuration for LLM
- Use Langchain & LangSmith for observability
- Here is some sample code

```python
# llm_handler.py

import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.callbacks import LangSmithCallbackHandler
from custom_llms import OllamaLLM  # Custom LLM wrapper for Ollama

def initialize_llm():
    """
    Initializes the LLM based on the LLM_PROVIDER environment variable.
    Supports 'openai' and 'ollama'.

    Returns:
        llm: An instance of an LLM compatible with LangChain.
    """
    # Load environment variables
    load_dotenv()
    llm_provider = os.getenv('LLM_PROVIDER', 'openai').lower()

    if llm_provider == 'openai':
        # Initialize OpenAI LLM
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

        llm = OpenAI(
            api_key=openai_api_key,
            model_name='gpt-3.5-turbo',
            callbacks=[LangSmithCallbackHandler()],
            temperature=0.7
        )
        print("Initialized OpenAI LLM.")
        return llm

    elif llm_provider == 'ollama':
        # Initialize custom Ollama LLM
        llm = OllamaLLM(
            model_name='qwen',  # Replace with your desired model
            base_url='http://localhost:11434',
            temperature=0.7,
            callbacks=[LangSmithCallbackHandler()]
        )
        print("Initialized Ollama LLM with model 'qwen'.")
        return llm

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER '{llm_provider}'. Please use 'openai' or 'ollama'.")
```

### 7.2.3 process each block, `processing.py`
- Generate title, summary, rewrite content, and tags for each article
- Here is some sample code

```python
# processing.py

from langchain import LLMChain
from prompts import title_prompt, summary_prompt, rewrite_prompt, tagging_prompt
from typing import List
from article import Article

def generate_title(llm, content: str) -> str:
    """
    Generates a new title for the article content using the LLM.

    Args:
        llm: The language model instance.
        content (str): The full text content of the article.

    Returns:
        str: The generated title.
    """
    chain = LLMChain(llm=llm, prompt=title_prompt)
    title = chain.run(content).strip()
    return title

def generate_summary(llm, content: str) -> str:
    """
    Generates a summary for the article content using the LLM.

    Args:
        llm: The language model instance.
        content (str): The full text content of the article.

    Returns:
        str: The generated summary.
    """
    chain = LLMChain(llm=llm, prompt=summary_prompt)
    summary = chain.run(content).strip()
    return summary

def rewrite_paragraph(llm, paragraph: str) -> str:
    """
    Rewrites a paragraph to be shorter without losing information.

    Args:
        llm: The language model instance.
        paragraph (str): The paragraph text to rewrite.

    Returns:
        str: The rewritten paragraph.
    """
    chain = LLMChain(llm=llm, prompt=rewrite_prompt)
    rewritten_paragraph = chain.run(paragraph).strip()
    return rewritten_paragraph

def generate_tags(llm, content: str) -> List[str]:
    """
    Generates tags for the article content using the LLM.

    Args:
        llm: The language model instance.
        content (str): The full text content of the article.

    Returns:
        List[str]: A list of tags assigned to the article.
    """
    chain = LLMChain(llm=llm, prompt=tagging_prompt)
    tags_response = chain.run(content)
    # Extract tags from the response
    tags = [tag.strip() for tag in tags_response.split('\n') if tag.strip().startswith('#')]
    return tags

def process_article(article: Article, llm) -> Article:
    """
    Processes an article by generating a new title, summary, rewritten content, and tags.

    Args:
        article (Article): The article object to process.
        llm: The language model instance.

    Returns:
        Article: The updated article object with new title, summary, content, and tags.
    """
    # Combine text content from content blocks
    content = ''
    for block in article.content_blocks:
        if block['type'] == 'paragraph':
            texts = block['paragraph']['text']
            paragraph_text = ''.join([text['plain_text'] for text in texts])
            content += paragraph_text + '\n'
        # You can include other block types if necessary

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
            # Reconstruct the block with the rewritten text
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
            # Preserve non-paragraph blocks (e.g., images)
            rewritten_blocks.append(block)

    # Update the article object
    article.title = new_title
    article.properties['Summary'] = summary
    article.properties['Tags'] = tags
    article.content_blocks = rewritten_blocks

    return article
```

### 7.2.4 `prompts.py`
- Define prompts for LLM
- Here is some sample code

```python
# prompts.py

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

### 7.2.5 `article.py`
- Define data model for article
- Here is some sample code

```python
# article.py

from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Article:
    """
    A data class representing an article from the Notion database.

    Attributes:
        page_id (str): The unique identifier of the Notion page.
        title (str): The title of the article.
        properties (Dict): A dictionary of the page's properties (e.g., tags, summary).
        content_blocks (List[Dict]): A list of content blocks (paragraphs, images, etc.) from the page.
    """
    page_id: str
    title: str
    properties: Dict = field(default_factory=dict)
    content_blocks: List[Dict] = field(default_factory=list)

    def get_full_content(self) -> str:
        """
        Combines all text content from the content blocks into a single string.

        Returns:
            str: The full text content of the article.
        """
        content = ''
        for block in self.content_blocks:
            if block['type'] == 'paragraph':
                texts = block['paragraph']['text']
                paragraph_text = ''.join([text['plain_text'] for text in texts])
                content += paragraph_text + '\n'
            # Include other text-containing blocks if necessary
        return content.strip()

    def update_content_blocks(self, new_blocks: List[Dict]):
        """
        Updates the content blocks of the article.

        Args:
            new_blocks (List[Dict]): The new list of content blocks.
        """
        self.content_blocks = new_blocks
```

### 7.2.6 `utils.py`
- Define utility functions
- Here is some sample code

```python
# utils.py

import logging
import time
import re
from typing import List, Any, Callable
from functools import wraps

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Configures logging for the application.

    Args:
        log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
        log_file (str, optional): The file to which logs should be written. If None, logs are printed to stdout.
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if log_file:
        logging.basicConfig(level=log_level, format=log_format, filename=log_file)
    else:
        logging.basicConfig(level=log_level, format=log_format)

def rate_limit(max_per_second):
    """
    Decorator to limit the rate of function calls.

    Args:
        max_per_second (float): The maximum number of function calls per second.

    Returns:
        Callable: The decorated function with rate limiting applied.
    """
    min_interval = 1.0 / float(max_per_second)

    def decorator(func: Callable):
        last_time_called = [0.0]

        @wraps(func)
        def rate_limited_function(*args, **kwargs):
            elapsed = time.perf_counter() - last_time_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_time_called[0] = time.perf_counter()
            return ret

        return rate_limited_function

    return decorator

def split_text_by_length(text: str, max_length: int) -> List[str]:
    """
    Splits a text into a list of substrings, each with a maximum length.

    Args:
        text (str): The text to split.
        max_length (int): The maximum length of each substring.

    Returns:
        List[str]: A list of substrings.
    """
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def sanitize_text(text: str) -> str:
    """
    Sanitizes text by removing unwanted characters or patterns.

    Args:
        text (str): The text to sanitize.

    Returns:
        str: The sanitized text.
    """
    # Example: Remove extra whitespace and control characters
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text.strip()

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Splits a list into smaller lists of a specified maximum size.

    Args:
        lst (List[Any]): The list to split.
        chunk_size (int): The maximum size of each chunk.

    Returns:
        List[List[Any]]: A list of list chunks.
    """
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def retry_on_exception(max_retries: int, exceptions: Any, delay: float = 0.0):
    """
    Decorator to retry a function call upon specified exceptions.

    Args:
        max_retries (int): Maximum number of retries.
        exceptions (Exception or Tuple[Exception]): Exceptions to catch and retry upon.
        delay (float): Delay between retries in seconds.

    Returns:
        Callable: The decorated function with retry logic applied.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logging.warning(f"Exception occurred: {e}. Retrying ({retries + 1}/{max_retries})...")
                    retries += 1
                    if delay > 0:
                        time.sleep(delay)
            # Last attempt
            return func(*args, **kwargs)
        return wrapped_function
    return decorator

def truncate_text(text: str, max_length: int) -> str:
    """
    Truncates text to a maximum length, adding an ellipsis if truncated.

    Args:
        text (str): The text to truncate.
        max_length (int): The maximum allowed length.

    Returns:
        str: The truncated text.
    """
    if len(text) <= max_length:
        return text
    else:
        return text[:max_length - 3] + '...'

def extract_tags_from_response(response: str) -> List[str]:
    """
    Extracts tags from the LLM response.

    Args:
        response (str): The LLM's response containing tags.

    Returns:
        List[str]: A list of extracted tags.
    """
    # Assume tags are lines starting with '#'
    tags = [line.strip() for line in response.splitlines() if line.strip().startswith('#')]
    return tags

def clean_llm_response(response: str) -> str:
    """
    Cleans up the LLM's response by removing unnecessary whitespace and artifacts.

    Args:
        response (str): The raw response from the LLM.

    Returns:
        str: The cleaned response.
    """
    return response.strip()
```

### 7.2.7 `app.py`
- Entry point of the application
- Here is some sample code 

``` python
# app.py

import os
import logging
from dotenv import load_dotenv
from notion_client import Client
from article import Article
from notion_client import fetch_articles, update_page_properties, update_page_content
from processing import process_article
from llm_handler import initialize_llm
from utils import setup_logging

def main():
    # Set up logging (logs to console; set log_file parameter to log to a file)
    setup_logging(log_level=logging.INFO)

    # Load environment variables from .env file
    load_dotenv()

    # Retrieve required environment variables
    notion_token = os.getenv('NOTION_TOKEN')
    database_id = os.getenv('DATABASE_ID')
    if not notion_token or not database_id:
        logging.error("Environment variables NOTION_TOKEN and DATABASE_ID must be set.")
        return

    # Initialize Notion client
    notion = Client(auth=notion_token)
    logging.info("Initialized Notion client.")

    # Initialize Language Model (LLM)
    try:
        llm = initialize_llm()
        logging.info("Initialized LLM.")
    except Exception as e:
        logging.error(f"Failed to initialize LLM: {e}")
        return

    # Fetch articles from Notion database
    try:
        articles = fetch_articles(notion, database_id)
        logging.info(f"Fetched {len(articles)} articles from the database.")
    except Exception as e:
        logging.error(f"Failed to fetch articles: {e}")
        return

    if not articles:
        logging.info("No articles found to process.")
        return

    # Process each article
    for article in articles:
        try:
            logging.info(f"Processing article {article.page_id}...")
            # Process the article (generate title, summary, rewrite content, generate tags)
            processed_article = process_article(article, llm)
            # Update the article's properties and content in Notion
            update_page_properties(notion, processed_article)
            update_page_content(notion, processed_article)
            logging.info(f"Successfully processed article {article.page_id}.")
        except Exception as e:
            logging.error(f"Error processing article {article.page_id}: {e}")

if __name__ == '__main__':
    main()
```
```

# 8. Additional Requirements

## 8.1. Data Model Specifications
### Source Database Fields
- Title: Text
- Content: Rich text blocks including paragraphs and images
- Original URL: URL
- Published Date: Date

### Destination Database Fields
- Title: Text (generated)
- Summary: Rich text
- Rewritten Content: Rich text blocks
- Tags: Multi-select from predefined options
- Original Article Reference: Relation to source database
- Processing Status: Select (Pending/Completed/Failed)

## 8.2. Error Handling
- Implement exponential backoff for API rate limits
- Retry failed operations up to 3 times
- Log all errors with stack traces
- Continue processing remaining articles if one fails
- Store error status in database for failed articles

## 8.3. Performance Requirements
- Process each article within 5 minutes
- Handle articles up to 10,000 words
- Implement timeout of 60 seconds for each LLM operation

## 8.4. Testing Requirements
### 8.4.1. Unit Testing
* Write unit tests for each function in processing.py, notion_client.py, and llm_handler.py.
* Use a testing framework like pytest.

### 8.4.2. Integration Testing
* Test the end-to-end processing of a single article.
* Verify that the article in Notion is updated correctly.


### 8.4.3. Quality Validation
- Generated titles must be under 100 characters
- Summaries must be one paragraph (150-300 words)
- Rewritten content must maintain all key information
- Tags must come from predefined list

### 8.4.4. Test Cases
- Short articles (<1000 words)
- Long articles (>5000 words)
- Articles with multiple images
- Articles with special characters/formatting
- Articles in different writing styles

## 8.5. Security
- Store all API keys in environment variables
- Log sensitive data only in debug mode
- Implement API key rotation mechanism
- Validate all input data before processing