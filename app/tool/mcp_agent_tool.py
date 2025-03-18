import asyncio
import json
from typing import Any, Dict, List, Optional

import aiohttp
from pydantic import Field, ConfigDict

from app.tool.base import BaseTool, ToolResult
from app.logger import logger


class MCPAgentTool(BaseTool):
    """
    Tool for interacting with an MCP (Model Context Protocol) agent.

    This tool connects to an MCP agent via API, allowing the agent to access
    external resources like Gmail, MongoDB, Typesense, etc. through a standardized
    protocol interface.
    """

    name: str = "mcp_agent"
    description: str = """
    Interact with an MCP (Model Context Protocol) agent to access external services and data sources.

    The MCP agent provides access to:
    - Gmail: Read, send, and manage emails
    - MongoDB: Query and manipulate data in MongoDB collections
    - Typesense: Perform search and indexing operations
    - And other integrated services

    This tool provides a standardized way to access these services through natural language.
    """

    parameters: dict = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The message to send to the MCP agent describing what you want to accomplish."
            },
            "context": {
                "type": "object",
                "description": "Optional context data to include with the request.",
                "additionalProperties": True
            }
        },
        "required": ["message"]
    }

    # MCP agent API endpoint
    api_url: str = "http://localhost:4013/v1.0/chat/mcp"
    session: Optional[aiohttp.ClientSession] = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def ensure_session(self):
        """Ensure an HTTP session exists for making requests."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def execute(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> ToolResult:
        """
        Execute the MCP agent tool by sending a request to the MCP agent API.

        Args:
            message: The natural language message to send to the MCP agent
            context: Optional context data to include with the request

        Returns:
            ToolResult: The result from the MCP agent
        """
        try:
            session = await self.ensure_session()

            payload = {
                "message": message,
                "userId": "63c6ac14fffcca1dec835575",
                "conversationId": "67d8f208d4854ef951bad1ab"
            }

            if context:
                payload["context"] = context

            logger.info(f"Sending request to MCP agent: {message[:100]}...")

            async with session.post(self.api_url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"MCP agent API error: Status {response.status}, {error_text}")
                    return ToolResult(error=f"MCP agent API error: {response.status} - {error_text}")

                result = await response.json()

                response_message = result.get("result", "")
                if not response_message:
                    logger.warning("MCP agent returned empty response")
                    return ToolResult(error="MCP agent returned empty response")

                logger.info(f"Received response from MCP agent: {response_message[:100]}...")
                return ToolResult(output=response_message)

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error connecting to MCP agent: {str(e)}")
            return ToolResult(error=f"HTTP error connecting to MCP agent: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing MCP agent response: {str(e)}")
            return ToolResult(error=f"Error parsing MCP agent response: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in MCP agent tool: {str(e)}")
            logger.exception("Detailed exception:")
            return ToolResult(error=f"Unexpected error: {str(e)}")

    async def cleanup(self):
        """Clean up resources when the tool is no longer needed."""
        if self.session:
            await self.session.close()
            self.session = None
