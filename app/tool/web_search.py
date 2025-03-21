import asyncio
from typing import List
import urllib.parse
from urllib.parse import urlparse

from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import config
from app.tool.base import BaseTool, ToolResult
from app.tool.search import (
    BaiduSearchEngine,
    DuckDuckGoSearchEngine,
    GoogleSearchEngine,
    WebSearchEngine,
)


class WebSearch(BaseTool):
    name: str = "web_search"
    description: str = """Perform a web search and return a list of relevant links.
    This function attempts to use the primary search engine API to get up-to-date results.
    If an error occurs, it falls back to an alternative search engine."""
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "(required) The search query to submit to the search engine.",
            },
            "num_results": {
                "type": "integer",
                "description": "(optional) The number of search results to return. Default is 10.",
                "default": 10,
            },
        },
        "required": ["query"],
    }
    _search_engine: dict[str, WebSearchEngine] = {
        "google": GoogleSearchEngine(),
        "baidu": BaiduSearchEngine(),
        "duckduckgo": DuckDuckGoSearchEngine(),
    }

    async def execute(self, query: str, num_results: int = 10) -> ToolResult:
        """
        Execute a Web search and return results with structured data.

        Args:
            query (str): The search query to submit to the search engine.
            num_results (int, optional): The number of search results to return. Default is 10.

        Returns:
            ToolResult: A ToolResult object containing structured search results.
        """
        engine_order = self._get_engine_order()

        for engine_name in engine_order:
            engine = self._search_engine[engine_name]
            try:
                links = await self._perform_search_with_engine(
                    engine, query, num_results
                )
                if links:
                    # Convert links to structured data
                    structured_results = []
                    for i, link in enumerate(links):
                        # Extract domain from URL
                        try:
                            parsed_url = urlparse(link)
                            domain = parsed_url.netloc

                            # Create result entry with structured data
                            result = {
                                "title": f"Result {i+1}",  # Default title if not available
                                "url": link,
                                "domain": domain,
                                "snippet": f"Found at {domain}",
                                "favicon": f"https://www.google.com/s2/favicons?domain={domain}"
                            }
                            structured_results.append(result)
                        except Exception as e:
                            # If parsing fails, add minimal info
                            structured_results.append({
                                "title": f"Result {i+1}",
                                "url": link,
                                "domain": "unknown",
                                "snippet": "No description available",
                                "favicon": ""
                            })

                    # Format the links for the text output
                    result_text = "\n".join([f"{i+1}. {link}" for i, link in enumerate(links)])

                    return ToolResult(
                        output=f"Search results for '{query}':\n{result_text}",
                        structured_data={
                            "query": query,
                            "engine": engine_name,
                            "results": structured_results,
                            "display_type": "search_results"
                        }
                    )
            except Exception as e:
                print(f"Search engine '{engine_name}' failed with error: {e}")

        # Return empty result if all engines fail
        return ToolResult(
            output=f"No search results found for '{query}'.",
            structured_data={
                "query": query,
                "results": [],
                "display_type": "search_results"
            }
        )

    def _get_engine_order(self) -> List[str]:
        """
        Determines the order in which to try search engines.
        Preferred engine is first (based on configuration), followed by the remaining engines.

        Returns:
            List[str]: Ordered list of search engine names.
        """
        preferred = "google"
        if config.search_config and config.search_config.engine:
            preferred = config.search_config.engine.lower()

        engine_order = []
        if preferred in self._search_engine:
            engine_order.append(preferred)
        for key in self._search_engine:
            if key not in engine_order:
                engine_order.append(key)
        return engine_order

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _perform_search_with_engine(
        self,
        engine: WebSearchEngine,
        query: str,
        num_results: int,
    ) -> List[str]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: list(engine.perform_search(query, num_results=num_results))
        )
