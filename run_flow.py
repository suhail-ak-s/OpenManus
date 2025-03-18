import asyncio
import time
import os

from app.agent.manus import Manus
from app.flow.base import FlowType
from app.flow.flow_factory import FlowFactory
from app.tool.focused_documentation_tool import FocusedDocumentationTool
from app.logger import logger


async def run_flow():
    # Create a project ID for documentation
    project_id = f"project_{int(time.time())}"

    # Create the Manus agent with focused documentation
    agent = Manus()
    agent.active_project_id = project_id

    # Add the agent to the agents dictionary
    agents = {
        "manus": agent,
    }

    try:
        prompt = input("Enter your prompt: ")

        if prompt.strip().isspace() or not prompt:
            logger.warning("Empty prompt provided.")
            return

        flow = FlowFactory.create_flow(
            flow_type=FlowType.PLANNING,
            agents=agents,
        )
        logger.warning("Processing your request...")

        # Log that we're using the focused documentation structure
        logger.info(f"Using focused documentation structure with project ID: {project_id}")
        logger.info(f"Documentation will be saved in: docs/{project_id}")

        try:
            start_time = time.time()
            result = await asyncio.wait_for(
                flow.execute(prompt),
                timeout=3600,  # 60 minute timeout for the entire execution
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Request processed in {elapsed_time:.2f} seconds")
            logger.info(result)

            # Print documentation summary
            project_dir = os.path.join("docs", project_id)
            if os.path.exists(project_dir):
                logger.info(f"Documentation generated: {project_dir}")
                steps_dir = os.path.join(project_dir, "steps")
                if os.path.exists(steps_dir):
                    step_dirs = [d for d in os.listdir(steps_dir) if os.path.isdir(os.path.join(steps_dir, d))]
                    logger.info(f"Total steps documented: {len(step_dirs)}")

        except asyncio.TimeoutError:
            logger.error("Request processing timed out after 1 hour")
            logger.info(
                "Operation terminated due to timeout. Please try a simpler request."
            )

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
    except Exception as e:
        logger.error(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(run_flow())
