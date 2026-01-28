"""
Project Architecture tool - Load codebase patterns and conventions.

Fetches architecture documentation, agent registry, and coding conventions
from the sentient-trader project to provide Claude Code with better context
when making implementation decisions.

Use before:
- Adding new agents
- Modifying trading logic
- Creating new MCP tools
- Refactoring existing code
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from mcp.types import TextContent
from pydantic import Field

from tools.shared.base_tool import BaseTool, BaseToolRequest

logger = logging.getLogger(__name__)

# Default project path
SENTIENT_TRADER_PATH = os.getenv(
    "SENTIENT_TRADER_PATH",
    "/Users/jackfelke/Projects/sentient-trader",
)


class ProjectArchRequest(BaseToolRequest):
    """Request model for project architecture tool."""

    aspect: str = Field(
        default="overview",
        description=(
            "Architecture aspect to fetch: "
            "'overview' (CLAUDE.md + high-level structure), "
            "'agents' (agent registry and patterns), "
            "'mcp' (MCP tool patterns), "
            "'prompts' (trading prompt patterns), "
            "'conventions' (naming and coding standards), "
            "'all' (comprehensive context)"
        ),
    )
    include_examples: bool = Field(
        default=True,
        description="Include code examples from the codebase",
    )


class ProjectArchTool(BaseTool):
    """
    Loads project architecture and conventions for development context.

    This tool reads local project files to provide Claude Code with
    understanding of:
    - Project structure and patterns
    - Agent architecture and registry
    - Naming conventions
    - MCP tool patterns
    - Prompt engineering patterns
    """

    name = "projectarch"
    description = (
        "Load sentient-trader architecture, patterns, and conventions. "
        "Use before adding agents, MCP tools, or modifying trading logic."
    )
    request_model = ProjectArchRequest

    async def execute(self, request: ProjectArchRequest) -> list[TextContent]:
        """Execute the architecture context load."""
        try:
            project_path = Path(SENTIENT_TRADER_PATH)
            if not project_path.exists():
                return [
                    TextContent(
                        type="text",
                        text=f"⚠️ Project not found at {SENTIENT_TRADER_PATH}\n"
                        "Set SENTIENT_TRADER_PATH environment variable.",
                    )
                ]

            context_parts = []

            if request.aspect in ("overview", "all"):
                context_parts.append(self._load_overview(project_path))

            if request.aspect in ("agents", "all"):
                context_parts.append(self._load_agents(project_path, request.include_examples))

            if request.aspect in ("mcp", "all"):
                context_parts.append(self._load_mcp_patterns(project_path, request.include_examples))

            if request.aspect in ("prompts", "all"):
                context_parts.append(self._load_prompt_patterns(project_path, request.include_examples))

            if request.aspect in ("conventions", "all"):
                context_parts.append(self._load_conventions(project_path))

            combined = "\n\n---\n\n".join(filter(None, context_parts))
            return [TextContent(type="text", text=combined)]

        except Exception as e:
            logger.error(f"Project architecture load failed: {e}")
            return [TextContent(type="text", text=f"⚠️ Failed to load architecture: {e}")]

    def _load_overview(self, project_path: Path) -> str:
        """Load project overview from CLAUDE.md."""
        claude_md = project_path / "CLAUDE.md"
        if not claude_md.exists():
            return "## Project Overview\n\n⚠️ CLAUDE.md not found"

        content = claude_md.read_text()
        # Extract key sections
        lines = ["## Project Overview\n"]

        # Get project vision section
        if "## Project Vision" in content:
            start = content.find("## Project Vision")
            end = content.find("##", start + 10)
            lines.append(content[start : end if end > start else start + 500])

        # Get architecture section
        if "## Architecture" in content:
            start = content.find("## Architecture")
            end = content.find("##", start + 10)
            lines.append(content[start : end if end > start else start + 500])

        # Get current status
        if "## Current Status" in content:
            start = content.find("## Current Status")
            end = content.find("##", start + 10)
            lines.append(content[start : end if end > start else start + 500])

        return "\n".join(lines)

    def _load_agents(self, project_path: Path, include_examples: bool) -> str:
        """Load agent registry and patterns."""
        registry_path = project_path / "src" / "multi_agent" / "registry.py"
        if not registry_path.exists():
            return "## Agent Architecture\n\n⚠️ Registry not found"

        lines = ["## Agent Architecture\n"]

        # Load registry file
        registry_content = registry_path.read_text()

        # Extract AGENT_REGISTRY entries
        lines.append("### Registered Agents\n")
        lines.append("```")

        # Find the AGENT_REGISTRY dict
        if "AGENT_REGISTRY" in registry_content:
            start = registry_content.find("AGENT_REGISTRY")
            # Get a reasonable chunk
            chunk = registry_content[start : start + 3000]
            lines.append(chunk[:2000])
        lines.append("```")

        if include_examples:
            # Show example agent structure
            sentinel_path = project_path / "src" / "multi_agent" / "sentinel.py"
            if sentinel_path.exists():
                lines.append("\n### Example Agent (market-scanner)\n")
                lines.append("```python")
                sentinel_content = sentinel_path.read_text()
                # Get class definition
                if "class MarketScanner" in sentinel_content:
                    start = sentinel_content.find("class MarketScanner")
                    lines.append(sentinel_content[start : start + 1500])
                lines.append("```")

        # Load naming conventions hook if exists
        hook_path = project_path / ".claude" / "hooks" / "naming-conventions.md"
        if hook_path.exists():
            lines.append("\n### Naming Conventions\n")
            lines.append(hook_path.read_text()[:1500])

        return "\n".join(lines)

    def _load_mcp_patterns(self, project_path: Path, include_examples: bool) -> str:
        """Load MCP tool patterns."""
        mcp_path = project_path / "src" / "mcp"
        if not mcp_path.exists():
            return "## MCP Tool Patterns\n\n⚠️ MCP directory not found"

        lines = ["## MCP Tool Patterns\n"]

        # List tool modules
        tools_path = mcp_path / "tools"
        if tools_path.exists():
            lines.append("### Tool Modules\n")
            for f in sorted(tools_path.glob("*.py")):
                if f.name != "__init__.py":
                    lines.append(f"- `{f.name}`")

        if include_examples:
            # Show example tool
            market_tools = tools_path / "market.py"
            if market_tools.exists():
                lines.append("\n### Example Tool (market.py)\n")
                lines.append("```python")
                content = market_tools.read_text()
                lines.append(content[:2000])
                lines.append("```")

        # Load tool registry if exists
        registry_path = mcp_path / "tool_registry.py"
        if registry_path.exists():
            lines.append("\n### Tool Registry Pattern\n")
            lines.append("```python")
            lines.append(registry_path.read_text()[:1500])
            lines.append("```")

        return "\n".join(lines)

    def _load_prompt_patterns(self, project_path: Path, include_examples: bool) -> str:
        """Load trading prompt patterns."""
        lines = ["## Trading Prompt Patterns\n"]

        # Check prompts locations
        prompts_path = project_path / "src" / "agent" / "prompts.py"
        brain_prompts = project_path / "src" / "brain" / "prompts"

        if prompts_path.exists():
            lines.append("### Main Trading Prompts\n")
            if include_examples:
                lines.append("```python")
                content = prompts_path.read_text()
                lines.append(content[:3000])
                lines.append("```")

        if brain_prompts.exists():
            lines.append("\n### Brain Prompts Directory\n")
            for f in sorted(brain_prompts.glob("*.py")):
                lines.append(f"- `{f.name}`")

        # Check for prompt examples in multi_agent
        sentinel_path = project_path / "src" / "multi_agent" / "sentinel.py"
        if sentinel_path.exists() and include_examples:
            content = sentinel_path.read_text()
            if "MARKET_SCANNER_PROMPT" in content:
                lines.append("\n### Example: Market Scanner Prompt\n")
                start = content.find("MARKET_SCANNER_PROMPT")
                lines.append("```python")
                lines.append(content[start : start + 2000])
                lines.append("```")

        return "\n".join(lines)

    def _load_conventions(self, project_path: Path) -> str:
        """Load naming and coding conventions."""
        lines = ["## Coding Conventions\n"]

        # Load from hooks
        hooks_path = project_path / ".claude" / "hooks"
        if hooks_path.exists():
            for hook in hooks_path.glob("*.md"):
                lines.append(f"\n### {hook.stem}\n")
                lines.append(hook.read_text()[:1000])

        # Add key patterns from CLAUDE.md
        claude_md = project_path / "CLAUDE.md"
        if claude_md.exists():
            content = claude_md.read_text()
            if "## Code Standards" in content:
                start = content.find("## Code Standards")
                end = content.find("##", start + 10)
                lines.append("\n### Code Standards (from CLAUDE.md)\n")
                lines.append(content[start : end if end > start else start + 1000])

        return "\n".join(lines)


# Export for tool registry
tool = ProjectArchTool()
