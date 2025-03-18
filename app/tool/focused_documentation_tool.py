import os
import json
import time
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from app.tool.base import BaseTool
from app.logger import logger


class DocumentationContent(BaseModel):
    """Model for documentation content to be saved"""
    content: str
    file_type: str = "md"
    title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FocusedDocumentationTool(BaseTool):
    """Tool for documenting agent's processes using a focused thought-action structure.

    This tool implements a hierarchical documentation approach where:
    - Each project contains steps
    - Each step contains thoughts
    - Each thought contains actions and resources

    This structure allows for more organized tracing of the agent's reasoning process
    and makes it easier to follow the relationship between thoughts and their resulting actions.
    """

    name: str = "focused_documentation"
    description: str = (
        "Document the agent's process, save content, and create structured documentation "
        "of the execution using a hierarchical thought-action approach. "
        "This helps maintain a clear record of the agent's reasoning and actions."
    )

    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": [
                    "init_project",
                    "document_plan",
                    "document_step",
                    "save_content",
                    "document_thought",
                    "document_action",
                    "save_urls"
                ],
                "description": "The documentation action to perform",
            },
            "project_id": {
                "type": "string",
                "description": "Unique identifier for the project",
            },
            "title": {
                "type": "string",
                "description": "Title or name for the project or document",
            },
            "content": {
                "type": "string",
                "description": "Content to document or save",
            },
            "content_type": {
                "type": "string",
                "enum": ["md", "html", "json", "txt"],
                "description": "Type of content being saved",
            },
            "step_index": {
                "type": "integer",
                "description": "Index of the current step in the plan",
            },
            "thought_index": {
                "type": "integer",
                "description": "Index of the thought within the current step",
            },
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of URLs to document",
            },
            "metadata": {
                "type": "object",
                "description": "Additional metadata to include with the documentation",
            }
        },
        "required": ["command", "project_id"],
    }

    base_dir: str = Field(default="docs")
    projects: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    thought_counters: Dict[str, Dict[int, int]] = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        os.makedirs(self.base_dir, exist_ok=True)
        self.thought_counters = {}

    async def execute(self, **kwargs) -> Any:
        """Execute the documentation tool with the given parameters"""
        command = kwargs.get("command")
        project_id = kwargs.get("project_id", f"project_{int(time.time())}")

        if not command:
            return {"error": "Command is required"}

        logger.info(f"DocumentationTool execute called with command: {command}, project_id: {project_id}")
        logger.info(f"Full kwargs: {kwargs}")
        logger.info(f"Will write to directory: {os.path.join(self.base_dir, project_id)}")

        try:
            if command == "init_project":
                result = await self._init_project(project_id, kwargs.get("title", "Untitled Project"))
                return result
            elif command == "document_plan":
                result = await self._document_plan(
                    project_id,
                    kwargs.get("content", ""),
                    kwargs.get("title", "Project Plan")
                )
                return result
            elif command == "document_step":
                return await self._document_step(
                    project_id,
                    kwargs.get("step_index", 0),
                    kwargs.get("content", ""),
                    kwargs.get("title", f"Step {kwargs.get('step_index', 0)}")
                )
            elif command == "document_thought":
                return await self._document_thought(
                    project_id,
                    kwargs.get("step_index", 0),
                    kwargs.get("content", "")
                )
            elif command == "document_action":
                return await self._document_action(
                    project_id,
                    kwargs.get("step_index", 0),
                    kwargs.get("content", ""),
                    kwargs.get("metadata", {})
                )
            elif command == "save_content":
                return await self._save_content(
                    project_id,
                    kwargs.get("step_index", 0),
                    kwargs.get("content", ""),
                    kwargs.get("content_type", "md"),
                    kwargs.get("title", "Content"),
                    kwargs.get("metadata", {})
                )
            elif command == "save_urls":
                return await self._save_urls(
                    project_id,
                    kwargs.get("step_index", 0),
                    kwargs.get("urls", []),
                    kwargs.get("title", "Browsing History")
                )
            else:
                return {"error": f"Unknown command: {command}"}
        except Exception as e:
            logger.error(f"Error in DocumentationTool: {str(e)}")
            return {"error": f"Documentation error: {str(e)}"}

    async def _init_project(self, project_id: str, title: str) -> Dict[str, Any]:
        """Initialize a new project with directories and basic structure"""
        project_dir = os.path.join(self.base_dir, project_id)

        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(os.path.join(project_dir, "steps"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "resources"), exist_ok=True)

        project_info = {
            "id": project_id,
            "title": title,
            "created_at": time.time(),
            "updated_at": time.time(),
            "steps": [],
            "resources": []
        }

        self.projects[project_id] = project_info
        with open(os.path.join(project_dir, "project.json"), "w") as f:
            json.dump(project_info, f, indent=2)

        readme_content = f"# {title}\n\nProject created at: {time.ctime()}\n\n## Overview\n\nThis documentation was automatically generated by the OpenManus AI agent.\n"
        with open(os.path.join(project_dir, "README.md"), "w") as f:
            f.write(readme_content)

        return {
            "success": True,
            "message": f"Project '{title}' initialized with ID: {project_id}",
            "project_id": project_id,
            "project_dir": project_dir
        }

    async def _document_plan(self, project_id: str, plan_content: str, title: str) -> Dict[str, Any]:
        """Document the initial plan for the project"""
        project_dir = os.path.join(self.base_dir, project_id)

        if not os.path.exists(project_dir):
            await self._init_project(project_id, title)

        plan_file = os.path.join(project_dir, "plan.md")

        formatted_content = f"""# {title}

## Generated Plan
Created at: {time.ctime()}

{plan_content}
"""

        with open(plan_file, "w") as f:
            f.write(formatted_content)

        if project_id in self.projects:
            self.projects[project_id]["plan"] = {
                "title": title,
                "created_at": time.time()
            }
            self.projects[project_id]["updated_at"] = time.time()

            with open(os.path.join(project_dir, "project.json"), "w") as f:
                json.dump(self.projects[project_id], f, indent=2)

        return {
            "success": True,
            "message": f"Plan documented for project {project_id}",
            "plan_file": plan_file
        }

    async def _document_step(self, project_id: str, step_index: int, content: str, title: str) -> Dict[str, Any]:
        """Document a step in the execution plan"""
        project_dir = os.path.join(self.base_dir, project_id)

        logger.info(f"_document_step called with: project_id={project_id}, step_index={step_index}, title={title}")
        logger.info(f"Project directory: {project_dir}")

        if not os.path.exists(project_dir):
            logger.warning(f"Project directory doesn't exist: {project_dir}")
            return {"error": f"Project {project_id} does not exist"}

        steps_dir = os.path.join(project_dir, "steps")
        if os.path.exists(steps_dir):
            existing_step_dirs = [d for d in os.listdir(steps_dir) if os.path.isdir(os.path.join(steps_dir, d))]
            logger.info(f"Existing step directories before creating new one: {existing_step_dirs}")
        else:
            logger.info("Steps directory doesn't exist yet, will create it")

        step_dir = os.path.join(project_dir, "steps", f"step_{step_index}")
        logger.info(f"Creating/updating step directory: {step_dir}")
        os.makedirs(step_dir, exist_ok=True)

        step_index_file = os.path.join(step_dir, "index.md")

        formatted_content = f"""# Step {step_index}: {title}

Started at: {time.ctime()}

## Description
{content}

## Execution Log
- Step initiated at {time.ctime()}
"""

        with open(step_index_file, "w") as f:
            f.write(formatted_content)

        if project_id in self.projects:
            if "steps" not in self.projects[project_id]:
                self.projects[project_id]["steps"] = []

            step_info = {
                "index": step_index,
                "title": title,
                "started_at": time.time(),
                "directory": step_dir
            }

            found = False
            for i, step in enumerate(self.projects[project_id]["steps"]):
                if step.get("index") == step_index:
                    self.projects[project_id]["steps"][i] = step_info
                    found = True
                    break

            if not found:
                self.projects[project_id]["steps"].append(step_info)

            self.projects[project_id]["updated_at"] = time.time()

            with open(os.path.join(project_dir, "project.json"), "w") as f:
                json.dump(self.projects[project_id], f, indent=2)

        if os.path.exists(steps_dir):
            existing_step_dirs = [d for d in os.listdir(steps_dir) if os.path.isdir(os.path.join(steps_dir, d))]
            logger.info(f"Existing step directories after creating new one: {existing_step_dirs}")

        return {
            "success": True,
            "message": f"Step {step_index} documented for project {project_id}",
            "step_dir": step_dir
        }

    async def _document_thought(self, project_id: str, step_index: int, content: str) -> Dict[str, Any]:
        """
        Document the agent's thought process for a step.

        Each thought is stored in its own directory structure with associated actions.
        Thoughts are numbered sequentially within each step.
        """
        project_dir = os.path.join(self.base_dir, project_id)
        step_dir = os.path.join(project_dir, "steps", f"step_{step_index}")

        if not os.path.exists(step_dir):
            await self._document_step(project_id, step_index, "Step execution", f"Step {step_index}")

        if project_id not in self.thought_counters:
            self.thought_counters[project_id] = {}

        if step_index not in self.thought_counters[project_id]:
            self.thought_counters[project_id][step_index] = 0

        thought_index = self.thought_counters[project_id][step_index]
        self.thought_counters[project_id][step_index] += 1

        thought_dir = os.path.join(step_dir, f"thought_{thought_index}")
        os.makedirs(thought_dir, exist_ok=True)

        thought_file = os.path.join(thought_dir, "thought.md")

        formatted_content = f"""## Agent's Thought Process #{thought_index}
Generated at: {time.ctime()}

{content}
"""

        with open(thought_file, "w") as f:
            f.write(formatted_content)

        step_index_file = os.path.join(step_dir, "index.md")
        if os.path.exists(step_index_file):
            with open(step_index_file, "a") as f:
                f.write(f"\n- Agent thought process #{thought_index} recorded at {time.ctime()}\n")

        actions_dir = os.path.join(thought_dir, "actions")
        os.makedirs(actions_dir, exist_ok=True)

        return {
            "success": True,
            "message": f"Thought process #{thought_index} documented for step {step_index}",
            "thought_file": thought_file,
            "thought_index": thought_index
        }

    async def _document_action(self, project_id: str, step_index: int, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Document an action taken by the agent.

        Actions are associated with specific thoughts and stored in the thought's directory.
        Each action is numbered sequentially within each thought.
        """
        project_dir = os.path.join(self.base_dir, project_id)
        step_dir = os.path.join(project_dir, "steps", f"step_{step_index}")

        if not os.path.exists(step_dir):
            await self._document_step(project_id, step_index, "Step execution", f"Step {step_index}")

        thought_index = 0
        if project_id in self.thought_counters and step_index in self.thought_counters[project_id]:
            thought_index = max(0, self.thought_counters[project_id][step_index] - 1)

        thought_dir = os.path.join(step_dir, f"thought_{thought_index}")

        if not os.path.exists(thought_dir):
            await self._document_thought(project_id, step_index, "Thought before action")
            thought_index = self.thought_counters[project_id][step_index] - 1
            thought_dir = os.path.join(step_dir, f"thought_{thought_index}")

        actions_dir = os.path.join(thought_dir, "actions")
        os.makedirs(actions_dir, exist_ok=True)

        existing_actions = [f for f in os.listdir(actions_dir) if f.startswith("action_") and f.endswith(".md")]
        action_index = len(existing_actions)

        action_file = os.path.join(actions_dir, f"action_{action_index}.md")

        formatted_content = f"""## Action #{action_index} Executed
Executed at: {time.ctime()}

### Tool Used
{metadata.get('tool_name', 'unknown')}

### Action Details
{content}

### Metadata
```json
{json.dumps(metadata, indent=2)}
```
"""

        with open(action_file, "w") as f:
            f.write(formatted_content)

        step_index_file = os.path.join(step_dir, "index.md")
        if os.path.exists(step_index_file):
            with open(step_index_file, "a") as f:
                f.write(f"\n- Agent performed action #{action_index} at {time.ctime()}\n")

        return {
            "success": True,
            "message": f"Action #{action_index} documented for step {step_index}, thought {thought_index}",
            "action_file": action_file,
            "thought_index": thought_index,
            "action_index": action_index
        }

    async def _save_content(self, project_id: str, step_index: int, content: str, content_type: str, title: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save general content as a resource for the project.

        This can be used for saving final results, intermediate findings, or any other content
        that doesn't fit into the thought-action model.

        Content is saved both in the step directory and in the project resources directory.
        """
        project_dir = os.path.join(self.base_dir, project_id)
        if not os.path.exists(project_dir):
            await self._init_project(project_id, "Untitled Project")

        thought_index = 0
        if project_id in self.thought_counters and step_index in self.thought_counters[project_id]:
            thought_index = max(0, self.thought_counters[project_id][step_index] - 1)

        safe_title = title.replace(" ", "_").replace("/", "_").replace("\\", "_")
        filename = f"{safe_title}_{int(time.time())}.{content_type}"

        resources_dir = os.path.join(project_dir, "resources")
        os.makedirs(resources_dir, exist_ok=True)
        resource_file = os.path.join(resources_dir, filename)

        with open(resource_file, "w") as f:
            f.write(content)

        if step_index != 999:
            step_dir = os.path.join(project_dir, "steps", f"step_{step_index}")

            if not os.path.exists(step_dir):
                await self._document_step(project_id, step_index, "Step with saved content", f"Step {step_index}")

            step_content_dir = os.path.join(step_dir, "content")
            os.makedirs(step_content_dir, exist_ok=True)

            step_content_file = os.path.join(step_content_dir, filename)
            with open(step_content_file, "w") as f:
                f.write(content)

            step_index_file = os.path.join(step_dir, "index.md")
            if os.path.exists(step_index_file):
                with open(step_index_file, "a") as f:
                    f.write(f"\n- Content '{title}' saved at {time.ctime()}\n")

            if thought_index is not None:
                thought_dir = os.path.join(step_dir, f"thought_{thought_index}")

                if not os.path.exists(thought_dir):
                    thought_result = await self._document_thought(project_id, step_index, f"Thought related to {title}")
                    thought_index = thought_result["thought_index"]
                    thought_dir = os.path.join(step_dir, f"thought_{thought_index}")

                thought_content_dir = os.path.join(thought_dir, "content")
                os.makedirs(thought_content_dir, exist_ok=True)

                thought_content_file = os.path.join(thought_content_dir, filename)
                with open(thought_content_file, "w") as f:
                    f.write(content)

                return {
                    "success": True,
                    "message": f"Content saved for step {step_index}, thought {thought_index}",
                    "content_file": thought_content_file,
                    "step_content_file": step_content_file,
                    "resource_file": resource_file,
                    "thought_index": thought_index
                }
        else:
            step_dir = os.path.join(project_dir, "steps", f"step_{step_index}")
            self._document_step(project_id, step_index, "Final Result", "Step 999")

            thought_result = await self._document_thought(project_id, step_index, f"Final Result")
            thought_index = thought_result["thought_index"]
            thought_dir = os.path.join(step_dir, f"thought_{thought_index}")

            thought_content_dir = os.path.join(thought_dir, "content")
            os.makedirs(thought_content_dir, exist_ok=True)

            thought_content_file = os.path.join(thought_content_dir, filename)
            with open(thought_content_file, "w") as f:
                f.write(content)

            step_content_dir = os.path.join(step_dir, "content")
            os.makedirs(step_content_dir, exist_ok=True)

            step_content_file = os.path.join(step_content_dir, filename)
            with open(step_content_file, "w") as f:
                f.write(content)

        return {
            "success": True,
            "message": f"Content saved for step {step_index}, thought {thought_index}",
            "content_file": thought_content_file if thought_index is not None else None,
            "step_content_file": step_content_file if step_index != 999 else None,
            "resource_file": resource_file,
            "thought_index": thought_index
        }

    async def _save_urls(self, project_id: str, step_index: int, urls: List[str], title: str) -> Dict[str, Any]:
        """
        Save a list of URLs as browsing history for the current thought.

        URLs are saved in both Markdown and JSON formats for better accessibility.
        """
        if not urls:
            return {
                "success": False,
                "message": "No URLs provided to save",
                "urls_count": 0
            }

        project_dir = os.path.join(self.base_dir, project_id)
        step_dir = os.path.join(project_dir, "steps", f"step_{step_index}")

        if not os.path.exists(step_dir):
            await self._document_step(project_id, step_index, "Step with browsing history", f"Step {step_index}")

        thought_index = 0
        if project_id in self.thought_counters and step_index in self.thought_counters[project_id]:
            thought_index = max(0, self.thought_counters[project_id][step_index] - 1)

        thought_dir = os.path.join(step_dir, f"thought_{thought_index}")

        if not os.path.exists(thought_dir):
            await self._document_thought(project_id, step_index, "Thought before browsing")
            thought_index = self.thought_counters[project_id][step_index] - 1
            thought_dir = os.path.join(step_dir, f"thought_{thought_index}")

        browsing_dir = os.path.join(thought_dir, "browsing")
        os.makedirs(browsing_dir, exist_ok=True)

        md_content = f"# {title}\n\nRecorded at: {time.ctime()}\n\n## URLs Visited\n\n"
        for i, url in enumerate(urls):
            parsed_url = urlparse(url)
            md_content += f"{i+1}. [{parsed_url.netloc}{parsed_url.path}]({url})\n"

        history_file = os.path.join(browsing_dir, "browsing_history.md")
        with open(history_file, "w") as f:
            f.write(md_content)

        history_json = {
            "title": title,
            "recorded_at": time.time(),
            "urls": urls
        }
        history_json_file = os.path.join(browsing_dir, "browsing_history.json")
        with open(history_json_file, "w") as f:
            json.dump(history_json, f, indent=2)

        step_index_file = os.path.join(step_dir, "index.md")
        if os.path.exists(step_index_file):
            with open(step_index_file, "a") as f:
                f.write(f"\n- Browsing history ({len(urls)} URLs) recorded at {time.ctime()}\n")

        return {
            "success": True,
            "message": f"Saved browsing history with {len(urls)} URLs for step {step_index}, thought {thought_index}",
            "history_file": history_file,
            "history_json_file": history_json_file
        }
