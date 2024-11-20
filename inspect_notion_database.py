import os
from notion_client import Client
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

NOTION_TOKEN = os.getenv('NOTION_TOKEN')
DATABASE_ID = os.getenv('SOURCE_DATABASE_ID')

# Initialize Notion client
notion = Client(auth=NOTION_TOKEN)

def get_database_properties(database_id):
    response = notion.databases.retrieve(database_id=database_id)
    return response


def get_database_pages(database_id):
    response = notion.databases.query(database_id=database_id)
    return response['results']

def get_page_blocks(page_id):
    blocks = []
    has_more = True
    next_cursor = None

    while has_more:
        response = notion.blocks.children.list(
            block_id=page_id,
            start_cursor=next_cursor
        )
        blocks.extend(response['results'])
        has_more = response['has_more']
        next_cursor = response.get('next_cursor')

    return blocks

def main():
    database_info = get_database_properties(DATABASE_ID)
    print("Database Properties:")
    print(json.dumps(database_info, indent=4, ensure_ascii=False))

    pages = get_database_pages(DATABASE_ID)
    if pages:
        first_page = pages[0]
        page_id = first_page['id']
        print("\nFirst Page Properties:")
        print(json.dumps(first_page['properties'], indent=4, ensure_ascii=False))

        page_blocks = get_page_blocks(page_id)
        print("\nFirst Page Content Blocks:")
        print(json.dumps(page_blocks, indent=4, ensure_ascii=False))
    else:
        print("No pages found in the database.")


if __name__ == '__main__':
    main()