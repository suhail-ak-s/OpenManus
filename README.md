# OpenManus Focused Documentation

This project implements a hierarchical documentation structure for AI agent processes, tracking steps, thoughts, and actions in a structured way.

## Core Components

### FocusedDocumentationTool

The `FocusedDocumentationTool` implements a hierarchical documentation approach for AI agent processes:

- **Projects**: Each task the agent works on gets its own project
- **Steps**: Projects are broken down into discrete steps
- **Thoughts**: Each step contains the agent's thoughts
- **Actions**: Thoughts lead to actions that the agent performs
- **Resources**: Content, code, links and other artifacts collected during execution

This structure creates a clear, navigable trail of the agent's thinking and actions.

## Key Files

- `app/tool/focused_documentation_tool.py`: The main implementation of the documentation tool
- `app/agent/manus.py`: An agent that uses the focused documentation tool
- `run_flow.py`: A script to run the agent with focused documentation

## Testing

Two test files are provided to verify the documentation structure:

- `focused_documentation_test.py`: Tests the documentation structure with synthetic data
- `real_world_documentation_test.py`: Tests the documentation structure with a real-world agent interaction

## Generated Documentation Structure

```
docs/
└── project_{timestamp}/
    ├── project.json           # Project metadata
    └── steps/
        └── step_{n}/          # Step directories
            ├── index.md       # Step overview
            └── thought_{n}/   # Thought directories
                ├── thought.md # Thought content
                └── actions/   # Action directory
                    ├── action_{n}.md  # Action files
                    └── resources/     # Resources directory
                        └── {filename} # Resource files
```

## Running

To use the focused documentation tool:

```
python run_flow.py
```

This will prompt you for a task and generate documentation in the `docs/` directory.
