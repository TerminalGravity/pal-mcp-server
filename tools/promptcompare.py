"""
Prompt Compare tool - Compare and analyze trading prompts with multiple models.

Uses multi-model consensus to evaluate prompt quality, identify improvements,
and test prompt variations. Helps optimize trading agent prompts.

Use cases:
- Compare two prompt versions for effectiveness
- Get multi-model feedback on prompt clarity
- Identify potential prompt improvements
- Test prompt with different model perspectives
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from mcp.types import TextContent
from pydantic import Field

from tools.shared.base_tool import BaseTool, BaseToolRequest

logger = logging.getLogger(__name__)

SENTIENT_TRADER_PATH = os.getenv(
    "SENTIENT_TRADER_PATH",
    "/Users/jackfelke/Projects/sentient-trader",
)


class PromptCompareRequest(BaseToolRequest):
    """Request model for prompt comparison tool."""

    mode: str = Field(
        default="analyze",
        description=(
            "Operation mode: "
            "'analyze' (analyze a single prompt), "
            "'compare' (compare two prompts), "
            "'list' (list available prompts), "
            "'suggest' (get improvement suggestions)"
        ),
    )
    prompt_name: str | None = Field(
        default=None,
        description="Name of prompt to analyze (e.g., 'MARKET_SCANNER_PROMPT', 'ANALYST_PROMPT')",
    )
    prompt_name_b: str | None = Field(
        default=None,
        description="Second prompt name for comparison mode",
    )
    custom_prompt: str | None = Field(
        default=None,
        description="Custom prompt text to analyze (instead of loading from file)",
    )
    focus: str = Field(
        default="trading",
        description="Analysis focus: 'trading' (trading-specific), 'clarity', 'completeness', 'safety'",
    )


class PromptCompareTool(BaseTool):
    """
    Analyzes and compares trading prompts for optimization.

    This tool helps improve trading agent prompts by:
    - Analyzing prompt structure and clarity
    - Comparing prompt versions
    - Suggesting improvements based on trading best practices
    """

    name = "promptcompare"
    description = "Analyze and compare trading prompts. " "Use to optimize agent prompts before deployment."
    request_model = PromptCompareRequest

    # Known prompt locations in sentient-trader
    PROMPT_LOCATIONS = {
        "MARKET_SCANNER_PROMPT": "src/multi_agent/sentinel.py",
        "ANALYST_PROMPT": "src/multi_agent/analyst_pkg/prompt.py",
        "SNIPER_PROMPT": "src/multi_agent/sniper.py",
        "EXIT_ANALYST_PROMPT": "src/multi_agent/exit_analyst.py",
        "THESIS_GUARD_PROMPT": "src/multi_agent/thesis_guard.py",
        "ARBITER_PROMPT": "src/multi_agent/arbiter.py",
        "AUTONOMY_PROMPT": "src/brain/autonomy_prompt.py",
        "DSPY_TRADER_PROMPT": "src/brain/dspy_trader.py",
    }

    async def execute(self, request: PromptCompareRequest) -> list[TextContent]:
        """Execute the prompt analysis."""
        try:
            project_path = Path(SENTIENT_TRADER_PATH)

            if request.mode == "list":
                return [TextContent(type="text", text=self._list_prompts(project_path))]

            if request.mode == "analyze":
                if request.custom_prompt:
                    prompt_text = request.custom_prompt
                    prompt_name = "Custom Prompt"
                elif request.prompt_name:
                    prompt_text = self._load_prompt(project_path, request.prompt_name)
                    prompt_name = request.prompt_name
                else:
                    return [TextContent(type="text", text="⚠️ Provide prompt_name or custom_prompt")]

                analysis = self._analyze_prompt(prompt_name, prompt_text, request.focus)
                return [TextContent(type="text", text=analysis)]

            if request.mode == "compare":
                if not request.prompt_name or not request.prompt_name_b:
                    return [TextContent(type="text", text="⚠️ Provide both prompt_name and prompt_name_b")]

                prompt_a = self._load_prompt(project_path, request.prompt_name)
                prompt_b = self._load_prompt(project_path, request.prompt_name_b)
                comparison = self._compare_prompts(
                    request.prompt_name,
                    prompt_a,
                    request.prompt_name_b,
                    prompt_b,
                    request.focus,
                )
                return [TextContent(type="text", text=comparison)]

            if request.mode == "suggest":
                if not request.prompt_name and not request.custom_prompt:
                    return [TextContent(type="text", text="⚠️ Provide prompt_name or custom_prompt")]

                if request.custom_prompt:
                    prompt_text = request.custom_prompt
                    prompt_name = "Custom Prompt"
                else:
                    prompt_text = self._load_prompt(project_path, request.prompt_name)
                    prompt_name = request.prompt_name

                suggestions = self._suggest_improvements(prompt_name, prompt_text, request.focus)
                return [TextContent(type="text", text=suggestions)]

            return [TextContent(type="text", text=f"⚠️ Unknown mode: {request.mode}")]

        except Exception as e:
            logger.error(f"Prompt analysis failed: {e}")
            return [TextContent(type="text", text=f"⚠️ Analysis failed: {e}")]

    def _list_prompts(self, project_path: Path) -> str:
        """List available prompts."""
        lines = ["## Available Trading Prompts\n"]

        for name, path in self.PROMPT_LOCATIONS.items():
            full_path = project_path / path
            exists = "✅" if full_path.exists() else "❌"
            lines.append(f"{exists} **{name}**")
            lines.append(f"   └─ `{path}`")

        lines.append("\n### Usage")
        lines.append("```")
        lines.append('promptcompare mode="analyze" prompt_name="MARKET_SCANNER_PROMPT"')
        lines.append('promptcompare mode="compare" prompt_name="ANALYST_PROMPT" prompt_name_b="EXIT_ANALYST_PROMPT"')
        lines.append("```")

        return "\n".join(lines)

    def _load_prompt(self, project_path: Path, prompt_name: str) -> str:
        """Load a prompt from the codebase."""
        if prompt_name not in self.PROMPT_LOCATIONS:
            return f"⚠️ Unknown prompt: {prompt_name}"

        file_path = project_path / self.PROMPT_LOCATIONS[prompt_name]
        if not file_path.exists():
            return f"⚠️ File not found: {file_path}"

        content = file_path.read_text()

        # Extract the prompt constant
        if prompt_name in content:
            start = content.find(prompt_name)
            # Find the string start (triple quote)
            quote_start = content.find('"""', start)
            if quote_start == -1:
                quote_start = content.find("'''", start)

            if quote_start > -1:
                quote_end = content.find('"""', quote_start + 3)
                if quote_end == -1:
                    quote_end = content.find("'''", quote_start + 3)
                if quote_end > -1:
                    return content[quote_start + 3 : quote_end].strip()

        return f"⚠️ Could not extract {prompt_name} from {file_path}"

    def _analyze_prompt(self, name: str, prompt: str, focus: str) -> str:
        """Analyze a single prompt."""
        lines = [f"## Prompt Analysis: {name}\n"]

        # Basic metrics
        word_count = len(prompt.split())
        line_count = len(prompt.split("\n"))
        char_count = len(prompt)

        lines.append("### Metrics")
        lines.append(f"- **Words:** {word_count}")
        lines.append(f"- **Lines:** {line_count}")
        lines.append(f"- **Characters:** {char_count}")

        # Structure analysis
        lines.append("\n### Structure")
        has_role = "role" in prompt.lower() or "you are" in prompt.lower()
        has_rules = "rule" in prompt.lower() or "must" in prompt.lower() or "never" in prompt.lower()
        has_examples = "example" in prompt.lower() or "e.g." in prompt.lower()
        has_output_format = "json" in prompt.lower() or "format" in prompt.lower() or "respond" in prompt.lower()

        lines.append(f"- Role Definition: {'✅' if has_role else '❌'}")
        lines.append(f"- Clear Rules: {'✅' if has_rules else '❌'}")
        lines.append(f"- Examples: {'✅' if has_examples else '⚠️ Consider adding'}")
        lines.append(f"- Output Format: {'✅' if has_output_format else '❌'}")

        if focus == "trading":
            lines.append("\n### Trading-Specific Analysis")
            has_risk = "risk" in prompt.lower() or "stop" in prompt.lower() or "loss" in prompt.lower()
            has_entry = "entry" in prompt.lower() or "enter" in prompt.lower()
            has_exit = "exit" in prompt.lower() or "close" in prompt.lower()
            has_confidence = "confidence" in prompt.lower() or "conviction" in prompt.lower()

            lines.append(f"- Risk Management: {'✅' if has_risk else '⚠️ Add risk guidance'}")
            lines.append(f"- Entry Criteria: {'✅' if has_entry else '⚠️ Define entry rules'}")
            lines.append(f"- Exit Criteria: {'✅' if has_exit else '⚠️ Define exit rules'}")
            lines.append(f"- Confidence Levels: {'✅' if has_confidence else '⚠️ Add confidence guidance'}")

        # Show prompt preview
        lines.append("\n### Prompt Preview (first 1000 chars)")
        lines.append("```")
        lines.append(prompt[:1000])
        if len(prompt) > 1000:
            lines.append(f"\n... ({len(prompt) - 1000} more characters)")
        lines.append("```")

        return "\n".join(lines)

    def _compare_prompts(self, name_a: str, prompt_a: str, name_b: str, prompt_b: str, focus: str) -> str:
        """Compare two prompts."""
        lines = [f"## Prompt Comparison: {name_a} vs {name_b}\n"]

        # Metrics comparison
        lines.append("### Metrics Comparison")
        lines.append("| Metric | " + name_a + " | " + name_b + " |")
        lines.append("|--------|-------|-------|")
        lines.append(f"| Words | {len(prompt_a.split())} | {len(prompt_b.split())} |")
        lines.append(f"| Lines | {len(prompt_a.split(chr(10)))} | {len(prompt_b.split(chr(10)))} |")
        lines.append(f"| Chars | {len(prompt_a)} | {len(prompt_b)} |")

        # Find common elements
        words_a = set(prompt_a.lower().split())
        words_b = set(prompt_b.lower().split())
        common = words_a & words_b
        unique_a = words_a - words_b
        unique_b = words_b - words_a

        lines.append("\n### Overlap Analysis")
        lines.append(f"- Common words: {len(common)}")
        lines.append(f"- Unique to {name_a}: {len(unique_a)}")
        lines.append(f"- Unique to {name_b}: {len(unique_b)}")

        # Key differences
        lines.append(f"\n### Key Unique Terms in {name_a}")
        key_terms_a = [w for w in unique_a if len(w) > 5][:10]
        lines.append(", ".join(key_terms_a) if key_terms_a else "None significant")

        lines.append(f"\n### Key Unique Terms in {name_b}")
        key_terms_b = [w for w in unique_b if len(w) > 5][:10]
        lines.append(", ".join(key_terms_b) if key_terms_b else "None significant")

        return "\n".join(lines)

    def _suggest_improvements(self, name: str, prompt: str, focus: str) -> str:
        """Suggest improvements for a prompt."""
        lines = [f"## Improvement Suggestions: {name}\n"]

        suggestions = []

        # Check for common issues
        if len(prompt) > 5000:
            suggestions.append("⚠️ **Long prompt** - Consider breaking into sections or extracting constants")

        if "you are" not in prompt.lower() and "role" not in prompt.lower():
            suggestions.append("💡 **Add role definition** - Start with 'You are a...' for clarity")

        if "example" not in prompt.lower():
            suggestions.append("💡 **Add examples** - Include 1-2 examples of expected input/output")

        if "json" not in prompt.lower() and "format" not in prompt.lower():
            suggestions.append("💡 **Specify output format** - Define expected response structure")

        if focus == "trading":
            if "risk" not in prompt.lower():
                suggestions.append("🎯 **Add risk guidance** - Include stop loss, position sizing rules")

            if "confidence" not in prompt.lower():
                suggestions.append("🎯 **Add confidence levels** - Define what high/medium/low confidence means")

            if "market" not in prompt.lower():
                suggestions.append("🎯 **Add market context** - Reference indicators, timeframes, conditions")

        if not suggestions:
            suggestions.append("✅ Prompt looks well-structured!")

        for s in suggestions:
            lines.append(f"- {s}")

        lines.append("\n### Quick Fixes Template")
        lines.append("```")
        lines.append("# Add to beginning:")
        lines.append("You are a [ROLE] responsible for [RESPONSIBILITY].")
        lines.append("")
        lines.append("# Add output format:")
        lines.append("Respond with JSON: {")
        lines.append('  "action": "...",')
        lines.append('  "confidence": 0.0-1.0,')
        lines.append('  "reasoning": "..."')
        lines.append("}")
        lines.append("```")

        return "\n".join(lines)


# Export for tool registry
tool = PromptCompareTool()
