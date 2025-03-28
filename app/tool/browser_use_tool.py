import asyncio
import json
from typing import Generic, Optional, TypeVar
import logging

from browser_use import Browser as BrowserUseBrowser
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.dom.service import DomService
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from app.config import config
from app.llm import LLM
from app.tool.base import BaseTool, ToolResult
from app.tool.web_search import WebSearch


_BROWSER_DESCRIPTION = """
Interact with a web browser to perform actions like navigation, interaction, content extraction, and tab management.

Key actions:
- Navigation: go_to_url, go_back, refresh, web_search
- Interaction: click_element, input_text, scroll_down/up, scroll_to_text, send_keys
- Content: extract_content, get_dropdown_options, select_dropdown_option
- Tabs: switch_tab, open_tab, close_tab
- Utility: wait

Use extract_content to retrieve specific information from pages, and web_search for focused search queries.
"""

Context = TypeVar("Context")


class BrowserUseTool(BaseTool, Generic[Context]):
    name: str = "browser_use"
    description: str = _BROWSER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "go_to_url",
                    "click_element",
                    "input_text",
                    "scroll_down",
                    "scroll_up",
                    "scroll_to_text",
                    "send_keys",
                    "get_dropdown_options",
                    "select_dropdown_option",
                    "go_back",
                    "web_search",
                    "wait",
                    "extract_content",
                    "switch_tab",
                    "open_tab",
                    "close_tab",
                ],
                "description": "The browser action to perform",
            },
            "url": {
                "type": "string",
                "description": "URL for 'go_to_url' or 'open_tab' actions",
            },
            "index": {
                "type": "integer",
                "description": "Element index for 'click_element', 'input_text', 'get_dropdown_options', or 'select_dropdown_option' actions",
            },
            "text": {
                "type": "string",
                "description": "Text for 'input_text', 'scroll_to_text', or 'select_dropdown_option' actions",
            },
            "scroll_amount": {
                "type": "integer",
                "description": "Pixels to scroll (positive for down, negative for up) for 'scroll_down' or 'scroll_up' actions",
            },
            "tab_id": {
                "type": "integer",
                "description": "Tab ID for 'switch_tab' action",
            },
            "query": {
                "type": "string",
                "description": "Search query for 'web_search' action",
            },
            "goal": {
                "type": "string",
                "description": "Extraction goal for 'extract_content' action",
            },
            "keys": {
                "type": "string",
                "description": "Keys to send for 'send_keys' action",
            },
            "seconds": {
                "type": "integer",
                "description": "Seconds to wait for 'wait' action",
            },
        },
        "required": ["action"],
        "dependencies": {
            "go_to_url": ["url"],
            "click_element": ["index"],
            "input_text": ["index", "text"],
            "switch_tab": ["tab_id"],
            "open_tab": ["url"],
            "scroll_down": ["scroll_amount"],
            "scroll_up": ["scroll_amount"],
            "scroll_to_text": ["text"],
            "send_keys": ["keys"],
            "get_dropdown_options": ["index"],
            "select_dropdown_option": ["index", "text"],
            "go_back": [],
            "web_search": ["query"],
            "wait": ["seconds"],
            "extract_content": ["goal"],
        },
    }

    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    browser: Optional[BrowserUseBrowser] = Field(default=None, exclude=True)
    context: Optional[BrowserContext] = Field(default=None, exclude=True)
    dom_service: Optional[DomService] = Field(default=None, exclude=True)
    web_search_tool: WebSearch = Field(default_factory=WebSearch, exclude=True)

    # Context for generic functionality
    tool_context: Optional[Context] = Field(default=None, exclude=True)

    llm: Optional[LLM] = Field(default_factory=LLM)

    @field_validator("parameters", mode="before")
    def validate_parameters(cls, v: dict, info: ValidationInfo) -> dict:
        if not v:
            raise ValueError("Parameters cannot be empty")
        return v

    async def _ensure_browser_initialized(self) -> BrowserContext:
        """Ensure browser and context are initialized."""
        if self.browser is None:
            browser_config_kwargs = {"headless": False, "disable_security": True}

            if config.browser_config:
                from browser_use.browser.browser import ProxySettings

                # handle proxy settings.
                if config.browser_config.proxy and config.browser_config.proxy.server:
                    browser_config_kwargs["proxy"] = ProxySettings(
                        server=config.browser_config.proxy.server,
                        username=config.browser_config.proxy.username,
                        password=config.browser_config.proxy.password,
                    )

                browser_attrs = [
                    "headless",
                    "disable_security",
                    "extra_chromium_args",
                    "chrome_instance_path",
                    "wss_url",
                    "cdp_url",
                ]

                for attr in browser_attrs:
                    value = getattr(config.browser_config, attr, None)
                    if value is not None:
                        if not isinstance(value, list) or value:
                            browser_config_kwargs[attr] = value

            self.browser = BrowserUseBrowser(BrowserConfig(**browser_config_kwargs))

        if self.context is None:
            context_config = BrowserContextConfig()

            # if there is context config in the config, use it.
            if (
                config.browser_config
                and hasattr(config.browser_config, "new_context_config")
                and config.browser_config.new_context_config
            ):
                context_config = config.browser_config.new_context_config

            self.context = await self.browser.new_context(context_config)
            self.dom_service = DomService(await self.context.get_current_page())

        return self.context

    async def execute(
        self,
        action: str,
        url: Optional[str] = None,
        index: Optional[int] = None,
        text: Optional[str] = None,
        scroll_amount: Optional[int] = None,
        tab_id: Optional[int] = None,
        query: Optional[str] = None,
        goal: Optional[str] = None,
        keys: Optional[str] = None,
        seconds: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """
        Execute a specified browser action.

        Args:
            action: The browser action to perform
            url: URL for navigation or new tab
            index: Element index for click or input actions
            text: Text for input action or search query
            scroll_amount: Pixels to scroll for scroll action
            tab_id: Tab ID for switch_tab action
            query: Search query for Google search
            goal: Extraction goal for content extraction
            keys: Keys to send for keyboard actions
            seconds: Seconds to wait
            **kwargs: Additional arguments

        Returns:
            ToolResult with the action's output or error
        """
        async with self.lock:
            try:
                context = await self._ensure_browser_initialized()

                # Get max content length from config
                max_content_length = getattr(
                    config.browser_config, "max_content_length", 2000
                )

                # Navigation actions
                if action == "go_to_url":
                    if not url:
                        return ToolResult(
                            error="URL is required for 'go_to_url' action"
                        )
                    page = await context.get_current_page()
                    await page.goto(url)
                    await page.wait_for_load_state()
                    return ToolResult(
                        output=f"Navigated to {url}",
                        structured_data={
                            "action_type": "navigate",
                            "url": url,
                            "display_type": "iframe"
                        }
                    )

                elif action == "go_back":
                    await context.go_back()
                    page = await context.get_current_page()
                    try:
                        current_url = page.url
                    except:
                        # Fallback to getting URL from state
                        try:
                            state = await context.get_state()
                            current_url = state.url
                        except:
                            current_url = "unknown_url"

                    return ToolResult(
                        output="Navigated back",
                        structured_data={
                            "action_type": "navigate_back",
                            "url": current_url,
                            "display_type": "iframe"
                        }
                    )

                elif action == "refresh":
                    await context.refresh_page()
                    page = await context.get_current_page()
                    try:
                        current_url = page.url
                    except:
                        # Fallback to getting URL from state
                        try:
                            state = await context.get_state()
                            current_url = state.url
                        except:
                            current_url = "unknown_url"

                    return ToolResult(
                        output="Refreshed current page",
                        structured_data={
                            "action_type": "refresh",
                            "url": current_url,
                            "display_type": "iframe"
                        }
                    )

                elif action == "web_search":
                    if not query:
                        return ToolResult(
                            error="Query is required for 'web_search' action"
                        )
                    search_results = await self.web_search_tool.execute(query)

                    if search_results:
                        # Navigate to the first search result
                        first_result = search_results[0]
                        if isinstance(first_result, dict) and "url" in first_result:
                            url_to_navigate = first_result["url"]
                        elif isinstance(first_result, str):
                            url_to_navigate = first_result
                        else:
                            return ToolResult(
                                error=f"Invalid search result format: {first_result}"
                            )

                        page = await context.get_current_page()
                        await page.goto(url_to_navigate)
                        await page.wait_for_load_state()

                        return ToolResult(
                            output=f"Searched for '{query}' and navigated to first result: {url_to_navigate}\nAll results:"
                            + "\n".join([str(r) for r in search_results]),
                            structured_data={
                                "action_type": "web_search",
                                "query": query,
                                "url": url_to_navigate,
                                "search_results": search_results,
                                "display_type": "iframe"
                            }
                        )
                    else:
                        return ToolResult(
                            error=f"No search results found for '{query}'"
                        )

                # Element interaction actions
                elif action == "click_element":
                    if index is None:
                        return ToolResult(
                            error="Index is required for 'click_element' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    download_path = await context._click_element_node(element)
                    output = f"Clicked element at index {index}"
                    if download_path:
                        output += f" - Downloaded file to {download_path}"
                    return ToolResult(
                        output=output,
                        structured_data={
                            "action_type": "click",
                            "element_index": index,
                            "download_path": download_path if download_path else None,
                            "display_type": "element_interaction"
                        }
                    )

                elif action == "input_text":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'input_text' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    await context._input_text_element_node(element, text)
                    return ToolResult(
                        output=f"Input '{text}' into element at index {index}",
                        structured_data={
                            "action_type": "input",
                            "element_index": index,
                            "text": text,
                            "display_type": "element_interaction"
                        }
                    )

                elif action == "scroll_down" or action == "scroll_up":
                    direction = 1 if action == "scroll_down" else -1
                    amount = (
                        scroll_amount
                        if scroll_amount is not None
                        else context.config.browser_window_size["height"]
                    )
                    await context.execute_javascript(
                        f"window.scrollBy(0, {direction * amount});"
                    )
                    return ToolResult(
                        output=f"Scrolled {'down' if direction > 0 else 'up'} by {amount} pixels",
                        structured_data={
                            "action_type": "scroll",
                            "direction": "down" if direction > 0 else "up",
                            "amount": amount,
                            "display_type": "scroll_indicator"
                        }
                    )

                elif action == "scroll_to_text":
                    if not text:
                        return ToolResult(
                            error="Text is required for 'scroll_to_text' action"
                        )
                    page = await context.get_current_page()
                    try:
                        locator = page.get_by_text(text, exact=False)
                        await locator.scroll_into_view_if_needed()
                        return ToolResult(
                            output=f"Scrolled to text: '{text}'",
                            structured_data={
                                "action_type": "scroll_to_text",
                                "text": text,
                                "display_type": "text_highlight"
                            }
                        )
                    except Exception as e:
                        return ToolResult(error=f"Failed to scroll to text: {str(e)}")

                elif action == "send_keys":
                    if not keys:
                        return ToolResult(
                            error="Keys are required for 'send_keys' action"
                        )
                    page = await context.get_current_page()
                    await page.keyboard.press(keys)
                    return ToolResult(
                        output=f"Sent keys: {keys}",
                        structured_data={
                            "action_type": "keys",
                            "keys": keys,
                            "display_type": "keyboard_interaction"
                        }
                    )

                elif action == "get_dropdown_options":
                    if index is None:
                        return ToolResult(
                            error="Index is required for 'get_dropdown_options' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    page = await context.get_current_page()
                    options = await page.evaluate(
                        """
                        (xpath) => {
                            const select = document.evaluate(xpath, document, null,
                                XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                            if (!select) return null;
                            return Array.from(select.options).map(opt => ({
                                text: opt.text,
                                value: opt.value,
                                index: opt.index
                            }));
                        }
                    """,
                        element.xpath,
                    )
                    return ToolResult(output=f"Dropdown options: {options}")

                elif action == "select_dropdown_option":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'select_dropdown_option' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    page = await context.get_current_page()
                    await page.select_option(element.xpath, label=text)
                    return ToolResult(
                        output=f"Selected option '{text}' from dropdown at index {index}"
                    )

                # Content extraction actions
                elif action == "extract_content":
                    if not goal:
                        return ToolResult(
                            error="Goal is required for 'extract_content' action"
                        )
                    page = await context.get_current_page()
                    try:
                        # Get page content and convert to markdown for better processing
                        html_content = await page.content()

                        # Get additional page information for visualization
                        page_title = await page.title()
                        current_url = page.url

                        # Take a screenshot for visualization
                        screenshot = None
                        try:
                            logging.info(f"Attempting to capture screenshot with full_page=True")
                            # Try different screenshot methods
                            try:
                                # Try using full page screenshot first
                                buffer = await page.screenshot(full_page=True)
                                import base64
                                screenshot = base64.b64encode(buffer).decode('utf-8')
                                logging.info(f"Successfully captured full page screenshot, size: {len(screenshot) if screenshot else 0} bytes")
                                # Ensure proper format for the base64 image
                                if screenshot:
                                    screenshot = f"data:image/png;base64,{screenshot}"
                            except Exception as full_page_error:
                                logging.error(f"Full page screenshot failed: {str(full_page_error)}")
                                # Try viewport screenshot as fallback
                                try:
                                    logging.info("Attempting to capture viewport screenshot")
                                    buffer = await page.screenshot()
                                    import base64
                                    screenshot = base64.b64encode(buffer).decode('utf-8')
                                    logging.info(f"Successfully captured viewport screenshot, size: {len(screenshot) if screenshot else 0} bytes")
                                    # Ensure proper format for the base64 image
                                    if screenshot:
                                        screenshot = f"data:image/png;base64,{screenshot}"
                                except Exception as ss_error:
                                    logging.error(f"Failed to capture viewport screenshot: {str(ss_error)}")
                        except Exception as screenshot_error:
                            logging.error(f"Failed to capture any screenshot: {str(screenshot_error)}")

                        # Import markdownify here to avoid global import
                        try:
                            import markdownify

                            content = markdownify.markdownify(html_content)
                        except ImportError:
                            # Fallback if markdownify is not available
                            content = html_content

                        # Create prompt for LLM
                        prompt_text = """
Your task is to extract the content of the page. You will be given a page and a goal, and you should extract all relevant information around this goal from the page.

Examples of extraction goals:
- Extract all company names
- Extract specific descriptions
- Extract all information about a topic
- Extract links with companies in structured format
- Extract all links

If the goal is vague, summarize the page. Respond in JSON format.

Extraction goal: {goal}

Page content:
{page}
"""
                        # Format the prompt with the goal and content
                        max_content_length = min(50000, len(content))
                        formatted_prompt = prompt_text.format(
                            goal=goal, page=content[:max_content_length]
                        )

                        # Create a proper message list for the LLM
                        from app.schema import Message

                        messages = [Message.user_message(formatted_prompt)]

                        # Use LLM to extract content based on the goal
                        response = await self.llm.ask(messages)

                        msg = f"Extracted from page:\n{response}\n"

                        # Get viewport and scroll position
                        viewport_info = None
                        scroll_position = None
                        try:
                            # Get viewport dimensions
                            viewport_info = await page.evaluate("""() => {
                                return {
                                    width: window.innerWidth,
                                    height: window.innerHeight
                                }
                            }""")

                            # Get scroll position
                            scroll_position = await page.evaluate("""() => {
                                return {
                                    x: window.scrollX,
                                    y: window.scrollY,
                                    max_y: Math.max(
                                        document.body.scrollHeight,
                                        document.documentElement.scrollHeight,
                                        document.body.offsetHeight,
                                        document.documentElement.offsetHeight
                                    )
                                }
                            }""")
                        except:
                            pass

                        # Create structured data
                        structured_data = {
                            "action_type": "extract_content",
                            "goal": goal,
                            "url": current_url,
                            "display_type": "content_extraction",
                            "extraction_result": response,
                            "page_details": {
                                "title": page_title,
                                "has_screenshot": screenshot is not None,
                                "viewport": viewport_info,
                                "scroll_position": scroll_position
                            }
                        }

                        # Return the extract content result with the page information
                        result = ToolResult(
                            output=msg,
                            structured_data=structured_data,
                            base64_image=screenshot
                        )

                        # Update the has_screenshot flag based on whether we have a screenshot
                        if hasattr(result, "structured_data") and result.structured_data:
                            if isinstance(result.structured_data, str):
                                try:
                                    data = json.loads(result.structured_data)
                                    data["has_screenshot"] = bool(screenshot)
                                    result.structured_data = json.dumps(data)
                                except:
                                    pass
                            elif isinstance(result.structured_data, dict):
                                result.structured_data["has_screenshot"] = bool(screenshot)

                        return result
                    except Exception as e:
                        # Provide a more helpful error message
                        error_msg = f"Failed to extract content: {str(e)}"
                        try:
                            # Try to get page details even during error
                            page_title = "Unknown Page"
                            current_url = "unknown_url"
                            screenshot = None
                            viewport_info = None
                            scroll_position = None

                            try:
                                page_title = await page.title()
                                current_url = page.url

                                # Improve screenshot capture with fallbacks
                                try:
                                    logging.info(f"[Error handler] Attempting to capture screenshot with full_page=True")
                                    # Try full page screenshot first
                                    buffer = await page.screenshot(full_page=True)
                                    import base64
                                    screenshot = base64.b64encode(buffer).decode('utf-8')
                                    logging.info(f"[Error handler] Successfully captured full page screenshot, size: {len(screenshot) if screenshot else 0} bytes")
                                    # Ensure proper format for the base64 image
                                    if screenshot:
                                        screenshot = f"data:image/png;base64,{screenshot}"
                                except Exception as full_page_error:
                                    logging.error(f"[Error handler] Full page screenshot failed: {str(full_page_error)}")
                                    # Try viewport screenshot as fallback
                                    try:
                                        logging.info("[Error handler] Attempting to capture viewport screenshot")
                                        buffer = await page.screenshot()
                                        import base64
                                        screenshot = base64.b64encode(buffer).decode('utf-8')
                                        logging.info(f"[Error handler] Successfully captured viewport screenshot, size: {len(screenshot) if screenshot else 0} bytes")
                                        # Ensure proper format for the base64 image
                                        if screenshot:
                                            screenshot = f"data:image/png;base64,{screenshot}"
                                    except Exception as ss_error:
                                        logging.error(f"[Error handler] Failed to capture viewport screenshot: {str(ss_error)}")

                                # Get viewport dimensions
                                viewport_info = await page.evaluate("""() => {
                                    return {
                                        width: window.innerWidth,
                                        height: window.innerHeight
                                    }
                                }""")

                                # Get scroll position
                                scroll_position = await page.evaluate("""() => {
                                    return {
                                        x: window.scrollX,
                                        y: window.scrollY,
                                        max_y: Math.max(
                                            document.body.scrollHeight,
                                            document.documentElement.scrollHeight,
                                            document.body.offsetHeight,
                                            document.documentElement.offsetHeight
                                        )
                                    }
                                }""")
                            except:
                                # Fallback to state info if available
                                try:
                                    state = await context.get_state()
                                    current_url = state.url
                                    page_title = state.title
                                except:
                                    pass

                            # Try to return a portion of the page content as fallback
                            return ToolResult(
                                output=f"{error_msg}\nHere's a portion of the page content:\n{content[:2000]}...",
                                base64_image=screenshot,
                                structured_data={
                                    "action_type": "extract_content",
                                    "goal": goal,
                                    "url": current_url,
                                    "display_type": "content_extraction",
                                    "extraction_result": f"{error_msg}\nHere's a portion of the page content:\n{content[:2000]}...",
                                    "error": True,
                                    "page_details": {
                                        "title": page_title,
                                        "has_screenshot": screenshot is not None,
                                        "viewport": viewport_info,
                                        "scroll_position": scroll_position
                                    }
                                }
                            )
                        except:
                            # If all else fails, just return the error
                            return ToolResult(
                                error=error_msg,
                                structured_data={
                                    "action_type": "extract_content",
                                    "goal": goal,
                                    "url": "unknown_url",
                                    "display_type": "content_extraction",
                                    "extraction_result": error_msg,
                                    "error": True,
                                    "page_details": {
                                        "title": "Unknown Page",
                                        "has_screenshot": False,
                                        "viewport": None,
                                        "scroll_position": None
                                    }
                                }
                            )

                # Tab management actions
                elif action == "switch_tab":
                    if tab_id is None:
                        return ToolResult(
                            error="Tab ID is required for 'switch_tab' action"
                        )
                    await context.switch_to_tab(tab_id)
                    page = await context.get_current_page()
                    await page.wait_for_load_state()
                    return ToolResult(output=f"Switched to tab {tab_id}")

                elif action == "open_tab":
                    if not url:
                        return ToolResult(error="URL is required for 'open_tab' action")
                    await context.create_new_tab(url)
                    return ToolResult(output=f"Opened new tab with {url}")

                elif action == "close_tab":
                    await context.close_current_tab()
                    return ToolResult(output="Closed current tab")

                # Utility actions
                elif action == "wait":
                    seconds_to_wait = seconds if seconds is not None else 3
                    await asyncio.sleep(seconds_to_wait)
                    return ToolResult(output=f"Waited for {seconds_to_wait} seconds")

                else:
                    return ToolResult(error=f"Unknown action: {action}")

            except Exception as e:
                return ToolResult(error=f"Browser action '{action}' failed: {str(e)}")

    async def get_current_state(
        self, context: Optional[BrowserContext] = None
    ) -> ToolResult:
        """
        Get the current browser state as a ToolResult.
        If context is not provided, uses self.context.
        """
        try:
            # Use provided context or fall back to self.context
            ctx = context or self.context
            if not ctx:
                return ToolResult(error="Browser context not initialized")

            state = await ctx.get_state()

            # Create a viewport_info dictionary if it doesn't exist
            viewport_height = 0
            if hasattr(state, "viewport_info") and state.viewport_info:
                viewport_height = state.viewport_info.height
            elif hasattr(ctx, "config") and hasattr(ctx.config, "browser_window_size"):
                viewport_height = ctx.config.browser_window_size.get("height", 0)

            # Take a screenshot for the state
            try:
                page = await ctx.get_current_page()
                buffer = await page.screenshot(full_page=True)
                import base64
                screenshot = base64.b64encode(buffer).decode('utf-8')
                screenshot = f"data:image/png;base64,{screenshot}"
            except Exception as e:
                logging.error(f"Error taking screenshot in get_current_state: {str(e)}")
                screenshot = None

            # Build the state info with all required fields
            state_info = {
                "url": state.url,
                "title": state.title,
                "tabs": [tab.model_dump() for tab in state.tabs],
                "help": "[0], [1], [2], etc., represent clickable indices corresponding to the elements listed. Clicking on these indices will navigate to or interact with the respective content behind them.",
                "interactive_elements": (
                    state.element_tree.clickable_elements_to_string()
                    if state.element_tree
                    else ""
                ),
                "scroll_info": {
                    "pixels_above": getattr(state, "pixels_above", 0),
                    "pixels_below": getattr(state, "pixels_below", 0),
                    "total_height": getattr(state, "pixels_above", 0)
                    + getattr(state, "pixels_below", 0)
                    + viewport_height,
                },
                "viewport_height": viewport_height,
            }

            return ToolResult(
                output=json.dumps(state_info, indent=4, ensure_ascii=False),
                base64_image=screenshot,
            )
        except Exception as e:
            return ToolResult(error=f"Failed to get browser state: {str(e)}")

    async def cleanup(self):
        """Clean up browser resources."""
        async with self.lock:
            if self.context is not None:
                await self.context.close()
                self.context = None
                self.dom_service = None
            if self.browser is not None:
                await self.browser.close()
                self.browser = None

    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        if self.browser is not None or self.context is not None:
            try:
                asyncio.run(self.cleanup())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.cleanup())
                loop.close()

    @classmethod
    def create_with_context(cls, context: Context) -> "BrowserUseTool[Context]":
        """Factory method to create a BrowserUseTool with a specific context."""
        tool = cls()
        tool.tool_context = context
        return tool
        """Clean up resources when this object is garbage collected."""
        try:
            if hasattr(self, 'browser') and self.browser is not None:
                logging.error("BrowserUseTool being garbage collected while browser is still open. This should be explicitly closed.")
                # Don't run an event loop here as it can cause issues if another one is already running
        except Exception as e:
            logging.error(f"Error during BrowserUseTool garbage collection: {e}")
