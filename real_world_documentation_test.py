import asyncio
import os
import time
from typing import Dict, List, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

from app.agent.manus import Manus
from app.tool.focused_documentation_tool import FocusedDocumentationTool

async def test_real_world_documentation():
    """
    Test the focused documentation structure with a real-world agent interaction.
    This simulates a multi-step process where an agent collects information about
    tourist areas in Tokyo.
    """
    project_id = f"project_{int(time.time())}"
    logging.info(f"Starting real-world documentation test with project ID: {project_id}")

    agent = Manus()

    agent.active_project_id = project_id

    try:
        agent.max_steps = 20
        result = await agent.run("Find information about the best areas to stay in Tokyo for tourists")

        logging.info(f"Agent execution completed in {time.time() - int(project_id.split('_')[-1]):.2f} seconds")
        logging.info(f"Result: {result[:500]}...")

        project_dir = os.path.join("docs", project_id)
        logging.info(f"Analyzing project directory: {project_dir}")

        steps_dir = os.path.join(project_dir, "steps")
        if os.path.exists(steps_dir):
            step_dirs = [d for d in os.listdir(steps_dir) if os.path.isdir(os.path.join(steps_dir, d))]
            logging.info(f"Found {len(step_dirs)} step directories: {step_dirs}")

            # Count thoughts and actions
            stats = {
                "steps": 0,
                "thoughts": 0,
                "actions": 0,
                "step_details": {}
            }

            for step_dir in step_dirs:
                step_path = os.path.join(steps_dir, step_dir)
                step_index = int(step_dir.split('_')[1])
                stats["steps"] += 1
                stats["step_details"][step_index] = {"thoughts": 0, "actions": 0}

                thought_dirs = [d for d in os.listdir(step_path) if d.startswith("thought_") and os.path.isdir(os.path.join(step_path, d))]
                stats["thoughts"] += len(thought_dirs)
                stats["step_details"][step_index]["thoughts"] = len(thought_dirs)

                for thought_dir in thought_dirs:
                    thought_path = os.path.join(step_path, thought_dir)
                    actions_dir = os.path.join(thought_path, "actions")
                    if os.path.exists(actions_dir):
                        action_files = [f for f in os.listdir(actions_dir) if f.endswith(".md")]
                        stats["actions"] += len(action_files)
                        stats["step_details"][step_index]["actions"] += len(action_files)

            logging.info(f"Documentation statistics: {stats}")

            # Print directory tree
            logging.info("Directory tree structure of the project:")
            await print_directory_structure(project_dir)

            # Sample a thought and action file to verify content
            try:
                sample_step_dir = os.path.join(steps_dir, step_dirs[0])
                thought_dirs = [d for d in os.listdir(sample_step_dir) if d.startswith("thought_") and os.path.isdir(os.path.join(sample_step_dir, d))]
                if thought_dirs:
                    sample_thought_dir = os.path.join(sample_step_dir, thought_dirs[0])
                    sample_thought_file = os.path.join(sample_thought_dir, "thought.md")

                    if os.path.exists(sample_thought_file):
                        with open(sample_thought_file, "r") as f:
                            sample_thought_content = f.read()
                        logging.info(f"Sample thought content:\n{sample_thought_content[:500]}")

                    # Sample an action file if available
                    actions_dir = os.path.join(sample_thought_dir, "actions")
                    if os.path.exists(actions_dir):
                        action_files = [f for f in os.listdir(actions_dir) if f.endswith(".md")]
                        if action_files:
                            sample_action_file = os.path.join(actions_dir, action_files[0])
                            with open(sample_action_file, "r") as f:
                                sample_action_content = f.read()
                            logging.info(f"Sample action content:\n{sample_action_content[:500]}")
            except Exception as e:
                logging.error(f"Error while sampling files: {str(e)}")

            logging.info(f"Test completed. Check the {project_dir} directory for results.")

        return "Real-world test completed successfully!"
    except Exception as e:
        logging.error(f"Error during test: {str(e)}")
        return f"Test failed: {str(e)}"

async def print_directory_structure(base_dir: str, indent: int = 0, max_depth: int = 5):
    """Print the directory structure in a tree-like format with depth limit."""
    if not os.path.exists(base_dir) or indent >= max_depth * 4:
        return

    logging.info(f"{' ' * indent}{os.path.basename(base_dir)}/")
    indent += 2

    for item in sorted(os.listdir(base_dir)):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            await print_directory_structure(item_path, indent, max_depth)
        else:
            logging.info(f"{' ' * indent}{item}")

if __name__ == "__main__":
    result = asyncio.run(test_real_world_documentation())
    logging.info(result)
