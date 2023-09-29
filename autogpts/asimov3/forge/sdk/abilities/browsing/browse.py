from bs4 import BeautifulSoup
import requests
from typing import Dict, List


from ..registry import ability

@ability(
    name="browse_web",
    description="Browse to a specific URL",
    parameters=[
        {
            "name": "url",
            "description": "URL to browse to",
            "type": "string",
            "required": True,
        }
    ],
    output_type="None",
)
async def browse_web(agent, task_id: str, url: str) -> None:
    """
    Browse to a specific URL
    """
    # Implementation goes here

@ability(
    name="search_web",
    description="Search the web for a specific query",
    parameters=[
        {
            "name": "query",
            "description": "Query to search for",
            "type": "string",
            "required": True,
        }
    ],
    output_type="list[str]",
)
async def search_web(agent, task_id: str, query: str) -> List[str]:
    """
    Search the web for a specific query
    """
    # Implementation goes here

@ability(
    name="scrape_web",
    description="Scrape data from a specific URL",
    parameters=[
        {
            "name": "url",
            "description": "URL to scrape data from",
            "type": "string",
            "required": True,
        },
        {
            "name": "html_element",
            "description": "Specific HTML element to scrape",
            "type": "string",
            "required": False,
        },
    ],
    output_type="dict",
)
async def scrape_web(agent, task_id: str, url: str, html_element: str = None) -> str:
    """
    Scrape data from a specific URL
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # This is a very basic example that just returns the title of the page
    # In a real-world application, you would extract the specific data you need here
    if html_element:
        return soup.find(html_element).string
    else:
        return str(soup)
