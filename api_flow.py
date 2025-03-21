import asyncio
import time
import os
import json
import uuid
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
import uvicorn

from app.agent.manus import Manus
from app.logger import logger

app = FastAPI(title="OpenManus API", description="API for OpenManus agent with SSE streaming")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

active_flows = {}

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    max_steps: Optional[int] = 20
    timeout_seconds: Optional[int] = 3600

class StreamRequest(BaseModel):
    session_id: Optional[str] = None
    query: Optional[str] = None
    include_past_events: Optional[bool] = True
    event_types: Optional[List[str]] = ["status", "step", "complete"]

class FlowResponse(BaseModel):
    session_id: str
    status: str
    result: Optional[str] = None
    error: Optional[str] = None
    project_id: Optional[str] = None
    elapsed_time: Optional[float] = None
    step_count: Optional[int] = None

class AbortResponse(BaseModel):
    session_id: str
    status: str
    message: str

@app.get("/")
async def get_home(request: Request):
    return JSONResponse(
        content={"message": "Visit /static/index.html to use the OpenManus API client"}
    )

@app.post("/api/query", response_model=FlowResponse)
async def create_query(request: QueryRequest):
    session_id = request.session_id or str(uuid.uuid4())

    project_id = f"project_{int(time.time())}"

    active_flows[session_id] = {
        "status": "initializing",
        "project_id": project_id,
        "query": request.query,
        "start_time": time.time(),
        "steps": [],
        "result": None,
        "error": None,
        "aborted": False  # Track whether this session has been aborted
    }

    logger.info(f"Creating asyncio task to process query for session {session_id}")
    asyncio.create_task(process_query(session_id, request.query, project_id, request.max_steps, request.timeout_seconds))
    logger.info(f"Created asyncio task for session {session_id}")

    return JSONResponse(
        status_code=202,
        content={
            "session_id": session_id,
            "status": "initializing",
            "project_id": project_id
        }
    )

@app.post("/api/abort/{session_id}", response_model=AbortResponse)
async def abort_session(session_id: str):
    if session_id not in active_flows:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    flow_data = active_flows[session_id]

    if flow_data["status"] in ["completed", "error", "aborted"]:
        return {
            "session_id": session_id,
            "status": flow_data["status"],
            "message": f"Session already in terminal state: {flow_data['status']}"
        }

    flow_data["aborted"] = True
    flow_data["status"] = "aborting"

    return {
        "session_id": session_id,
        "status": "aborting",
        "message": "Abort signal sent. Session will terminate at next processing step."
    }

@app.get("/api/status/{session_id}", response_model=FlowResponse)
async def get_status(session_id: str):
    if session_id not in active_flows:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    flow_data = active_flows[session_id]

    elapsed_time = None
    if flow_data["start_time"]:
        if flow_data["status"] in ["completed", "error", "aborted"]:
            elapsed_time = flow_data.get("elapsed_time")
        else:
            elapsed_time = time.time() - flow_data["start_time"]

    return {
        "session_id": session_id,
        "status": flow_data["status"],
        "result": flow_data.get("result"),
        "error": flow_data.get("error"),
        "project_id": flow_data.get("project_id"),
        "elapsed_time": elapsed_time,
        "step_count": len(flow_data.get("steps", []))
    }

@app.get("/api/stream/{session_id}")
async def stream_results_get(request: Request, session_id: str):
    if session_id not in active_flows:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return EventSourceResponse(event_generator(request, session_id))

@app.post("/api/stream")
async def stream_results_post(request: Request, stream_request: StreamRequest, background_tasks: BackgroundTasks):
    session_id = stream_request.session_id
    query = stream_request.query

    if query and not session_id:
        session_id = str(uuid.uuid4())

        project_id = f"project_{int(time.time())}"

        active_flows[session_id] = {
            "status": "initializing",
            "project_id": project_id,
            "query": query,
            "start_time": time.time(),
            "steps": [],
            "result": None,
            "error": None,
            "aborted": False
        }

        # Directly start processing as an asyncio task instead of using BackgroundTasks
        # This ensures it starts immediately and doesn't wait for the response to complete
        logger.info(f"Creating asyncio task to process query for session {session_id}")
        asyncio.create_task(process_query(session_id, query, project_id, 20, 3600))
        logger.info(f"Created asyncio task for session {session_id}")

    if not session_id or session_id not in active_flows:
        if not session_id:
            raise HTTPException(status_code=400, detail="No session_id or query provided")
        else:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return EventSourceResponse(
        event_generator(
            request,
            session_id,
            include_past_events=stream_request.include_past_events,
            event_types=stream_request.event_types
        )
    )

async def event_generator(
    request: Request,
    session_id: str,
    include_past_events: bool = True,
    event_types: List[str] = ["status", "step", "complete"]
):
    flow_data = active_flows[session_id]

    if "status" in event_types:
        yield {
            "event": "status",
            "data": json.dumps({
                "status": flow_data["status"],
                "session_id": session_id,
                "project_id": flow_data.get("project_id")
            })
        }

    sent_steps = set()
    last_status = flow_data["status"]

    logger.info(f"Starting event generator for session {session_id}")

    # If include_past_events is True, send all existing steps immediately
    if include_past_events and "step" in event_types:
        steps = flow_data.get("steps", [])
        for step_idx, step in enumerate(steps):
            if step is not None:
                sent_steps.add(step_idx)

                step_type = step.get("type", "unknown")
                step_content = step.get("content", "")
                tool_name = step.get("tool", "")
                cycle_index = step.get("cycle_index", 0)
                structured_data = step.get("structured_data")
                base64_image = step.get("base64_image")

                if step_type == "thought":
                    title = f"Step {step_idx}: Thinking"
                    markdown = step_content
                elif step_type == "action":
                    title = f"Using {tool_name}"
                    markdown = f"Tool: {tool_name}\nArguments: {extract_args_from_content(step_content)}"
                elif step_type == "result":
                    title = f"Result from {tool_name}"
                    markdown = step_content
                else:
                    title = f"Step {step_idx}: {step.get('title', 'Unknown')}"
                    markdown = step_content

                logger.info(f"Sending existing step {step_idx} for session {session_id}")
                yield {
                    "event": "step",
                    "data": json.dumps({
                        "step_index": step_idx,
                        "title": title,
                        "content": markdown,
                        "type": step_type,
                        "cycle_index": cycle_index,
                        "tool": tool_name,
                        "step_info": step,
                        "structured_data": structured_data,
                        "base64_image": base64_image
                    })
                }

    # Continue streaming new events
    while True:
        if await request.is_disconnected():
            logger.info(f"Client disconnected from SSE stream for session {session_id}")
            break

        flow_data = active_flows[session_id]
        current_status = flow_data["status"]

        # Send status update if changed and status events are requested
        if current_status != last_status and "status" in event_types:
            last_status = current_status
            logger.info(f"Sending status update: {current_status} for session {session_id}")
            yield {
                "event": "status",
                "data": json.dumps({
                    "status": current_status,
                    "session_id": session_id
                })
            }

        # Get steps in order and send as numbered sequence
        if "step" in event_types:
            steps = flow_data.get("steps", [])

            for step_idx, step in enumerate(steps):
                if step_idx not in sent_steps and step is not None:
                    sent_steps.add(step_idx)

                    step_type = step.get("type", "unknown")
                    step_content = step.get("content", "")
                    tool_name = step.get("tool", "")
                    cycle_index = step.get("cycle_index", 0)
                    structured_data = step.get("structured_data")
                    base64_image = step.get("base64_image")

                    if step_type == "thought":
                        title = f"Step {step_idx}: Thinking"
                        markdown = step_content
                    elif step_type == "action":
                        title = f"Using {tool_name}"
                        markdown = f"Tool: {tool_name}\nArguments: {extract_args_from_content(step_content)}"
                    elif step_type == "result":
                        title = f"Result from {tool_name}"
                        markdown = step_content
                    else:
                        title = f"Step {step_idx}: {step.get('title', 'Unknown')}"
                        markdown = step_content

                    logger.info(f"Sending step {step_idx} for session {session_id}")
                    yield {
                        "event": "step",
                        "data": json.dumps({
                            "step_index": step_idx,
                            "title": title,
                            "content": markdown,
                            "type": step_type,
                            "cycle_index": cycle_index,
                            "tool": tool_name,
                            "step_info": step,
                            "structured_data": structured_data,
                            "base64_image": base64_image
                        })
                    }

        # Send complete event if the session is in a terminal state
        if current_status in ["completed", "error", "aborted"] and "complete" in event_types:
            logger.info(f"Sending complete event with status {current_status} for session {session_id}")

            final_result = flow_data.get("result", "")
            error_message = flow_data.get("error", "")

            if current_status == "completed":
                markdown_result = f"## ✅ Task Completed\n\n{final_result}"
            elif current_status == "aborted":
                markdown_result = f"## 🛑 Task Aborted\n\nThe task was aborted by user request."
            else:
                markdown_result = f"## ❌ Error Encountered\n\n{error_message}"

            yield {
                "event": "complete",
                "data": json.dumps({
                    "status": current_status,
                    "result": flow_data.get("result"),
                    "error": flow_data.get("error"),
                    "elapsed_time": flow_data.get("elapsed_time"),
                    "project_id": flow_data.get("project_id"),
                    "session_id": session_id,
                    "markdown": markdown_result
                })
            }
            break

        await asyncio.sleep(0.5)

def extract_args_from_content(content):
    if "Arguments:" in content:
        parts = content.split("Arguments:")
        if len(parts) > 1:
            return parts[1].strip()
    return content

def get_descriptive_title(content, cycle_index):
    lines = content.split('\n')

    for line in lines:
        if line.strip().startswith('#'):
            return line.strip('#').strip()

    first_sentence = content.split('.')[0] if '.' in content else content
    if len(first_sentence) > 50:
        first_sentence = first_sentence[:47] + "..."

    if "research" in first_sentence.lower():
        return f"Research travel information"
    elif "plan" in first_sentence.lower():
        return f"Plan travel itinerary"
    elif "explore" in first_sentence.lower():
        return f"Explore travel options"
    elif "japan" in first_sentence.lower():
        return f"Japan travel research"

    tasks = [
        "Research Japan travel basics",
        "Create travel preparation checklist",
        "Research cities and attractions",
        "Plan transportation options",
        "Explore accommodation choices",
        "Investigate dining options"
    ]

    return tasks[cycle_index % len(tasks)]

def get_first_line(content):
    lines = [line.strip() for line in content.split('\n') if line.strip()]

    for line in lines:
        if not line.startswith('#'):
            if len(line) > 100:
                return line[:97] + "..."
            return line

    if lines:
        return lines[0].lstrip('#').strip()

    return "Researching information..."

def get_action_display(tool_name, content):
    if tool_name == "web_search":
        search_term = ""
        if "search_term" in content:
            parts = content.split("search_term")
            if len(parts) > 1 and ":" in parts[1]:
                search_term = parts[1].split(":", 1)[1].strip().strip('"').strip("'")

        if search_term:
            title = f"Searching {search_term[:30]}{'...' if len(search_term) > 30 else ''}"
        else:
            title = "Searching the web"
        return "search", title

    elif tool_name == "browser_use":
        if "scroll" in content:
            return "browser", "Scrolling down"
        elif "get_text" in content:
            return "browser", "Reading webpage content"
        elif "navigate" in content:
            url = ""
            if "url" in content:
                parts = content.split("url")
                if len(parts) > 1 and ":" in parts[1]:
                    url = parts[1].split(":", 1)[1].strip().strip('"').strip("'").strip(",")

            if url:
                domain = url.replace("https://", "").replace("http://", "").split("/")[0]
                return "browser", f"Browsing {domain}"
            return "browser", "Browsing website"

        return "browser", "Interacting with website"

    elif tool_name == "run_terminal_cmd":
        cmd = ""
        if "command" in content:
            parts = content.split("command")
            if len(parts) > 1 and ":" in parts[1]:
                cmd = parts[1].split(":", 1)[1].strip().strip('"').strip("'")

        if cmd:
            if cmd.startswith("mkdir"):
                return "terminal", f"Executing command {cmd[:40]}{'...' if len(cmd) > 40 else ''}"
            elif cmd.startswith("cd"):
                return "terminal", f"Changing directory to {cmd[3:].strip()}"
            elif cmd.startswith("touch"):
                return "terminal", f"Creating file {cmd[6:].strip()}"
            else:
                return "terminal", f"Running command {cmd[:40]}{'...' if len(cmd) > 40 else ''}"

        return "terminal", "Executing command"

    return "tool", f"Using {tool_name}"

async def process_query(session_id: str, query: str, project_id: str, max_steps: int = 20, timeout_seconds: int = 3600):
    try:
        logger.info(f"STARTED PROCESSING: session {session_id} with query: '{query}'")
        logger.info(f"Process query function is running in task: {asyncio.current_task().get_name()}")

        active_flows[session_id]["status"] = "starting"
        logger.info(f"Updated status for session {session_id} to 'starting'")

        logger.info(f"Initializing Manus agent for session {session_id}")
        try:
            agent = Manus()
            logger.info(f"Agent instance created for session {session_id}")

            agent.active_project_id = project_id
            logger.info(f"Set project_id {project_id} for agent in session {session_id}")

            agent.max_steps = max_steps
            logger.info(f"Set max_steps {max_steps} for agent in session {session_id}")

            logger.info(f"Manus agent fully initialized for session {session_id}")
        except Exception as e:
            logger.error(f"Error initializing agent: {str(e)}")
            logger.exception("Agent initialization exception:")
            raise

        current_step = 0
        cycle_index = 0
        thought_action_map = {}

        original_think = agent.think
        original_act = agent.act

        async def tracked_think():
            nonlocal current_step, cycle_index
            logger.info(f"tracked_think called for session {session_id}, step {current_step}, cycle {cycle_index}")

            # Check for abort signal before processing
            if session_id in active_flows and active_flows[session_id].get("aborted", False):
                logger.info(f"Aborting session {session_id} during think phase")
                raise asyncio.CancelledError("Session aborted by user")

            result = await original_think()
            logger.info(f"Original think completed for session {session_id}")

            thought_content = None
            if agent.messages and len(agent.messages) > 0:
                last_message = agent.messages[-1]
                thought_content = last_message.content

            if session_id in active_flows and thought_content:
                # Check again for abort after potentially long LLM call
                if active_flows[session_id].get("aborted", False):
                    logger.info(f"Aborting session {session_id} after think phase")
                    raise asyncio.CancelledError("Session aborted by user")

                safe_content = str(thought_content) if thought_content else "Agent thinking..."

                step_info = {
                    "index": current_step,
                    "cycle_index": cycle_index,
                    "title": f"Thinking",
                    "content": safe_content[:500] + "..." if len(safe_content) > 500 else safe_content,
                    "timestamp": time.time(),
                    "type": "thought",
                    "parent_id": None,
                    "step_id": f"thought_{cycle_index}"
                }

                while len(active_flows[session_id].get("steps", [])) <= current_step:
                    active_flows[session_id].setdefault("steps", []).append(None)

                active_flows[session_id]["steps"][current_step] = step_info
                active_flows[session_id]["status"] = "processing"

                thought_action_map[cycle_index] = current_step

                current_step += 1
                logger.info(f"Added thought for cycle {cycle_index} as step {current_step-1} to session {session_id}")

            return result

        async def tracked_act():
            nonlocal current_step, cycle_index
            logger.info(f"tracked_act called for session {session_id}, step {current_step}, cycle {cycle_index}")

            # Check for abort signal before processing
            if session_id in active_flows and active_flows[session_id].get("aborted", False):
                logger.info(f"Aborting session {session_id} during act phase")
                raise asyncio.CancelledError("Session aborted by user")

            current_tool_calls = agent.tool_calls.copy() if agent.tool_calls else []

            result = await original_act()
            logger.info(f"Original act completed for session {session_id}")

            # Check again for abort after potentially long tool call
            if session_id in active_flows and active_flows[session_id].get("aborted", False):
                logger.info(f"Aborting session {session_id} after act phase")
                raise asyncio.CancelledError("Session aborted by user")

            if session_id in active_flows and current_tool_calls:
                tool_call = current_tool_calls[0]
                if tool_call and tool_call.function:
                    try:
                        import json
                        tool_name = tool_call.function.name
                        args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

                        args_str = json.dumps(args, indent=2)

                        action_content = f"Tool: {tool_name}\nArguments: {args_str}"

                        action_structured_data = None

                        if tool_name == "browser_use":
                            action_structured_data = {}
                            action_type = args.get("action", "")

                            if action_type == "go_to_url":
                                action_structured_data = {
                                    "action_type": "navigate",
                                    "url": args.get("url", ""),
                                    "display_type": "iframe"
                                }
                            elif action_type in ["scroll_down", "scroll_up"]:
                                direction = "down" if action_type == "scroll_down" else "up"
                                amount = args.get("scroll_amount", 0)
                                action_structured_data = {
                                    "action_type": "scroll",
                                    "direction": direction,
                                    "amount": amount,
                                    "display_type": "scroll_indicator"
                                }
                            elif action_type == "click_element":
                                action_structured_data = {
                                    "action_type": "click",
                                    "element_index": args.get("index", 0)
                                }
                            elif action_type == "extract_content":
                                action_structured_data = {
                                    "action_type": "extract_content",
                                    "goal": args.get("goal", ""),
                                    "display_type": "content_extraction"
                                }
                        elif tool_name == "web_search":
                            action_structured_data = {
                                "action_type": "web_search",
                                "query": args.get("query", ""),
                                "display_type": "search_query"
                            }

                        parent_step_index = thought_action_map.get(cycle_index)

                        step_info = {
                            "index": current_step,
                            "cycle_index": cycle_index,
                            "title": f"Using {tool_name}",
                            "content": action_content,
                            "timestamp": time.time(),
                            "type": "action",
                            "tool": tool_name,
                            "parent_id": f"thought_{cycle_index}",
                            "step_id": f"action_{cycle_index}",
                            "structured_data": action_structured_data  # Add structured data to the step info
                        }

                        while len(active_flows[session_id].get("steps", [])) <= current_step:
                            active_flows[session_id].setdefault("steps", []).append(None)

                        active_flows[session_id]["steps"][current_step] = step_info
                        active_flows[session_id]["status"] = "processing"

                        logger.info(f"Added action for cycle {cycle_index} as step {current_step} to session {session_id}")

                        current_step += 1

                        result_preview = str(result)
                        if len(result_preview) > 300:
                            result_preview = result_preview[:300] + "..."

                        result_structured_data = None
                        result_base64_image = None

                        if hasattr(agent, "_last_tool_result") and agent._last_tool_result:
                            tool_result = agent._last_tool_result
                            if hasattr(tool_result, "structured_data") and tool_result.structured_data:
                                result_structured_data = tool_result.structured_data
                                logger.info(f"Got structured_data from tool_result")

                            if hasattr(tool_result, "base64_image") and tool_result.base64_image:
                                result_base64_image = tool_result.base64_image
                                logger.info(f"Got base64_image from tool_result")

                        if not result_structured_data:
                            if hasattr(result, "structured_data") and result.structured_data:
                                result_structured_data = result.structured_data
                            elif tool_name == "browser_use" and action_structured_data:
                                if action_structured_data.get("action_type") == "navigate":
                                    result_structured_data = {
                                        "url": action_structured_data.get("url", ""),
                                        "display_type": "iframe"
                                    }
                                elif action_structured_data.get("action_type") == "extract_content":
                                    json_content = None
                                    if "```json" in result_preview:
                                        try:
                                            import re
                                            json_match = re.search(r"```json\s*([\s\S]*?)\s*```", result_preview)
                                            if json_match:
                                                json_str = json_match.group(1).strip()
                                                json_content = json.loads(json_str)
                                        except:
                                            pass

                                    result_structured_data = {
                                        "action_type": "extract_content",
                                        "goal": action_structured_data.get("goal", ""),
                                        "display_type": "content_extraction",
                                        "extraction_result": json_content or result_preview
                                    }
                                elif action_structured_data.get("action_type") == "web_search":
                                    result_structured_data = {
                                        "action_type": "web_search",
                                        "query": action_structured_data.get("query", ""),
                                        "display_type": "search_results"
                                    }

                        if not result_base64_image and hasattr(agent, "_current_base64_image") and agent._current_base64_image:
                            result_base64_image = agent._current_base64_image
                            logger.info(f"Got base64_image from _current_base64_image")

                        result_info = {
                            "index": current_step,
                            "cycle_index": cycle_index,
                            "title": f"Result from {tool_name}",
                            "content": result_preview,
                            "timestamp": time.time(),
                            "type": "result",
                            "tool": tool_name,
                            "parent_id": f"action_{cycle_index}",
                            "step_id": f"result_{cycle_index}",
                            "structured_data": result_structured_data,
                            "base64_image": result_base64_image
                        }

                        while len(active_flows[session_id].get("steps", [])) <= current_step:
                            active_flows[session_id].setdefault("steps", []).append(None)

                        active_flows[session_id]["steps"][current_step] = result_info
                        logger.info(f"Added result for cycle {cycle_index} as step {current_step} to session {session_id}")

                        current_step += 1
                        cycle_index += 1
                    except Exception as e:
                        logger.error(f"Error tracking action for cycle {cycle_index}: {str(e)}")
                        cycle_index += 1

            return result

        agent.think = tracked_think
        agent.act = tracked_act

        active_flows[session_id]["status"] = "processing"
        logger.info(f"Processing query for session {session_id}: {query}")

        start_time = time.time()

        try:
            # Monitor the abort flag during execution
            async def run_with_abort_check():
                while True:
                    if active_flows[session_id].get("aborted", False):
                        raise asyncio.CancelledError("Session aborted by user")
                    await asyncio.sleep(0.5)

            # Create a task for the agent execution
            logger.info(f"Creating agent execution task for session {session_id}")
            agent_task = asyncio.create_task(agent.run(query))
            logger.info(f"Created agent task for session {session_id}")

            # Create a task to monitor the abort flag
            abort_task = asyncio.create_task(run_with_abort_check())
            logger.info(f"Created abort monitor task for session {session_id}")

            # Wait for either the agent to finish or abort to be triggered
            logger.info(f"Waiting for tasks to complete for session {session_id}")
            done, pending = await asyncio.wait(
                [agent_task, abort_task],
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED
            )
            logger.info(f"Task waiting completed for session {session_id}, done tasks: {len(done)}, pending tasks: {len(pending)}")

            # Cancel any pending tasks
            for task in pending:
                logger.info(f"Cancelling pending task for session {session_id}")
                task.cancel()

            # If agent_task is done, get the result
            if agent_task in done:
                logger.info(f"Agent task completed for session {session_id}")
                result = agent_task.result()
            else:
                # If abort_task completed first, it means the abort flag was set
                logger.info(f"Agent task was not in done tasks for session {session_id}")
                result = "Operation aborted by user"
                active_flows[session_id]["status"] = "aborted"
                logger.info(f"Session {session_id} aborted by user")

            elapsed_time = time.time() - start_time

            # If we got here normally (not via abort)
            if active_flows[session_id]["status"] != "aborted":
                active_flows[session_id]["status"] = "completed"

            active_flows[session_id]["result"] = result
            active_flows[session_id]["elapsed_time"] = elapsed_time

            logger.info(f"Request processed in {elapsed_time:.2f} seconds")
            logger.info(f"Total steps: {current_step}")

            return result

        except asyncio.CancelledError:
            # This can be triggered by the abort signal
            logger.info(f"Session {session_id} cancelled")

            active_flows[session_id]["status"] = "aborted"
            active_flows[session_id]["result"] = "Operation aborted by user"
            active_flows[session_id]["elapsed_time"] = time.time() - start_time

            return "Operation aborted by user"

        except asyncio.TimeoutError:
            error_message = "Request processing timed out"
            logger.error(f"{error_message} after {timeout_seconds} seconds")

            active_flows[session_id]["status"] = "error"
            active_flows[session_id]["error"] = error_message
            active_flows[session_id]["elapsed_time"] = time.time() - start_time

            return None
        except Exception as e:
            error_message = f"Error during execution: {str(e)}"
            logger.error(error_message)
            logger.exception("Detailed exception:")

            active_flows[session_id]["status"] = "error"
            active_flows[session_id]["error"] = error_message
            active_flows[session_id]["elapsed_time"] = time.time() - start_time

            return None

    except Exception as e:
        error_message = f"Error processing query: {str(e)}"
        logger.error(error_message)
        logger.exception("Detailed exception:")

        if session_id in active_flows:
            active_flows[session_id]["status"] = "error"
            active_flows[session_id]["error"] = error_message
            if "start_time" in active_flows[session_id]:
                active_flows[session_id]["elapsed_time"] = time.time() - active_flows[session_id]["start_time"]

        return None

@app.get("/api/info")
def read_root():
    return {
        "api": "OpenManus API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "API info and welcome page"},
            {"path": "/static/index.html", "method": "GET", "description": "Web client for interacting with the API"},
            {"path": "/api/query", "method": "POST", "description": "Submit a new query to process"},
            {"path": "/api/status/{session_id}", "method": "GET", "description": "Get status of a query processing session"},
            {"path": "/api/stream/{session_id}", "method": "GET", "description": "Stream results using SSE (legacy)"},
            {"path": "/api/stream", "method": "POST", "description": "Stream results using SSE with configuration options"},
            {"path": "/api/abort/{session_id}", "method": "POST", "description": "Abort a running query"}
        ]
    }

if __name__ == "__main__":
    uvicorn.run("api_flow:app", host="0.0.0.0", port=8000, reload=True)
