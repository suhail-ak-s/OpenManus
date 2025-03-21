from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class BaseTool(ABC, BaseModel):
    name: str
    description: str
    parameters: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True

    async def __call__(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        return await self.execute(**kwargs)

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""

    def to_param(self) -> Dict:
        """Convert tool to function call format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolResult(BaseModel):
    """Represents the result of a tool execution."""

    output: Any = Field(default=None)
    error: Optional[str] = Field(default=None)
    base64_image: Optional[str] = Field(default=None)
    system: Optional[str] = Field(default=None)
    structured_data: Optional[Dict[str, Any]] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __bool__(self):
        return any(getattr(self, field) for field in self.__fields__)

    def __add__(self, other: "ToolResult"):
        def combine_fields(
            field: Optional[str], other_field: Optional[str], concatenate: bool = True
        ):
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("Cannot combine tool results")
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
            structured_data=combine_fields(self.structured_data, other.structured_data, False),
        )

    def __str__(self):
        """String representation that doesn't include base64_image data"""
        if self.error:
            return f"Error: {self.error}"

        # Don't include base64_image in the string representation
        # Focus on the text output and structured data if available
        if self.structured_data:
            import json

            # If output exists, combine with structured data summary
            if self.output:
                # Truncate structured_data in string representation to avoid huge outputs
                data_summary = str(self.structured_data)
                if len(data_summary) > 100:
                    data_summary = f"{data_summary[:97]}..."

                return f"{self.output}\n[Structured data available: {data_summary}]"

            # Return just structured data if no output
            try:
                return json.dumps(self.structured_data, indent=2)
            except:
                return str(self.structured_data)

        # Return just the output if no structured data
        return str(self.output) if self.output is not None else ""

    def replace(self, **kwargs):
        """Returns a new ToolResult with the given fields replaced."""
        # return self.copy(update=kwargs)
        return type(self)(**{**self.dict(), **kwargs})


class CLIResult(ToolResult):
    """A ToolResult that can be rendered as a CLI output."""


class ToolFailure(ToolResult):
    """A ToolResult that represents a failure."""
