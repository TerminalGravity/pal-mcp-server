#!/usr/bin/env python3
"""
Trading Development MCP Server

Standalone FastMCP server providing development tools for sentient-trader.
Runs separately from the main PAL MCP server.

Tools:
- tradingcontext: Live system state (positions, trades, market)
- projectarch: Architecture, patterns, conventions
- promptcompare: Prompt analysis and comparison

Usage:
    python trading_dev_server.py
"""

import logging
import os
from pathlib import Path

import httpx
from fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SENTIENT_TRADER_PATH = os.getenv(
    "SENTIENT_TRADER_PATH",
    "/Users/jackfelke/Projects/sentient-trader",
)
BLOFIN_MCP_URL = os.getenv(
    "BLOFIN_MCP_URL",
    "https://sentient-trader.fly.dev/mcp",
)

mcp = FastMCP(
    name="trading-dev",
    instructions="Development tools for sentient-trader - context, architecture, prompts",
)


# =============================================================================
# TRADING CONTEXT TOOL
# =============================================================================


@mcp.tool()
async def tradingcontext(
    context_type: str = "full",
    symbol: str = "BTC-USDT",
    trade_limit: int = 10,
) -> dict:
    """
    Fetch live trading system state for development context.

    Args:
        context_type: Type of context - full, positions, trades, market, agents, performance
        symbol: Trading symbol (default: BTC-USDT)
        trade_limit: Number of recent trades to include

    Returns:
        Current trading system state
    """
    try:
        results = {}

        async with httpx.AsyncClient(timeout=30.0) as client:
            if context_type in ("full", "positions"):
                try:
                    resp = await client.post(
                        f"{BLOFIN_MCP_URL}/tools/get_positions",
                        json={"symbol": symbol},
                    )
                    results["positions"] = resp.json() if resp.status_code == 200 else {"error": resp.text}
                except Exception as e:
                    results["positions"] = {"error": str(e)}

            if context_type in ("full", "trades"):
                try:
                    resp = await client.post(
                        f"{BLOFIN_MCP_URL}/tools/get_db_trades",
                        json={"symbol": symbol, "limit": trade_limit},
                    )
                    results["trades"] = resp.json() if resp.status_code == 200 else {"error": resp.text}
                except Exception as e:
                    results["trades"] = {"error": str(e)}

            if context_type in ("full", "market"):
                try:
                    resp = await client.post(
                        f"{BLOFIN_MCP_URL}/tools/get_market_snapshot",
                        json={"symbol": symbol},
                    )
                    results["market"] = resp.json() if resp.status_code == 200 else {"error": resp.text}
                except Exception as e:
                    results["market"] = {"error": str(e)}

            if context_type in ("full", "agents"):
                try:
                    resp = await client.post(
                        f"{BLOFIN_MCP_URL}/tools/get_agent_states",
                        json={},
                    )
                    results["agents"] = resp.json() if resp.status_code == 200 else {"error": resp.text}
                except Exception as e:
                    results["agents"] = {"error": str(e)}

            if context_type == "performance":
                try:
                    resp = await client.post(
                        f"{BLOFIN_MCP_URL}/tools/get_session_stats",
                        json={},
                    )
                    results["performance"] = resp.json() if resp.status_code == 200 else {"error": resp.text}
                except Exception as e:
                    results["performance"] = {"error": str(e)}

        return {"context_type": context_type, "symbol": symbol, "data": results}

    except Exception as e:
        return {"error": str(e), "hint": f"Ensure blofin-trading MCP is running at {BLOFIN_MCP_URL}"}


# =============================================================================
# PROJECT ARCHITECTURE TOOL
# =============================================================================


@mcp.tool()
def projectarch(
    aspect: str = "overview",
    include_examples: bool = True,
) -> dict:
    """
    Load project architecture, patterns, and conventions.

    Args:
        aspect: What to load - overview, agents, mcp, prompts, conventions, all
        include_examples: Include code examples

    Returns:
        Architecture documentation and code patterns
    """
    project_path = Path(SENTIENT_TRADER_PATH)
    if not project_path.exists():
        return {"error": f"Project not found at {SENTIENT_TRADER_PATH}"}

    results = {}

    if aspect in ("overview", "all"):
        claude_md = project_path / "CLAUDE.md"
        if claude_md.exists():
            content = claude_md.read_text()
            # Extract key sections
            sections = {}
            for section in ["Project Vision", "Architecture", "Current Status"]:
                if f"## {section}" in content:
                    start = content.find(f"## {section}")
                    end = content.find("##", start + 10)
                    sections[section] = content[start : end if end > start else start + 500]
            results["overview"] = sections

    if aspect in ("agents", "all"):
        registry_path = project_path / "src" / "multi_agent" / "registry.py"
        if registry_path.exists():
            content = registry_path.read_text()
            # Extract agent names
            import re

            agents = re.findall(r'"([a-z-]+)":\s*AgentSpec', content)
            results["agents"] = {
                "registered": agents,
                "count": len(agents),
            }

            if include_examples:
                # Get example agent
                sentinel_path = project_path / "src" / "multi_agent" / "sentinel.py"
                if sentinel_path.exists():
                    results["agents"]["example_file"] = str(sentinel_path)
                    results["agents"]["example_preview"] = sentinel_path.read_text()[:2000]

    if aspect in ("mcp", "all"):
        mcp_path = project_path / "src" / "mcp" / "tools"
        if mcp_path.exists():
            tools = [f.stem for f in mcp_path.glob("*.py") if f.name != "__init__.py"]
            results["mcp_tools"] = tools

    if aspect in ("prompts", "all"):
        prompts = {}
        # Check common prompt locations
        for name, path in [
            ("main_prompts", "src/agent/prompts.py"),
            ("brain_prompts", "src/brain/prompts"),
            ("autonomy", "src/brain/autonomy_prompt.py"),
        ]:
            full_path = project_path / path
            if full_path.exists():
                if full_path.is_file():
                    prompts[name] = full_path.read_text()[:3000] if include_examples else str(full_path)
                else:
                    prompts[name] = [f.name for f in full_path.glob("*.py")]
        results["prompts"] = prompts

    if aspect in ("conventions", "all"):
        hooks_path = project_path / ".claude" / "hooks"
        if hooks_path.exists():
            hooks = {}
            for hook in hooks_path.glob("*.md"):
                hooks[hook.stem] = hook.read_text()[:1000]
            results["conventions"] = hooks

    return {"aspect": aspect, "project_path": str(project_path), "data": results}


# =============================================================================
# PROMPT COMPARE TOOL
# =============================================================================

PROMPT_LOCATIONS = {
    "MARKET_SCANNER_PROMPT": "src/multi_agent/sentinel.py",
    "ANALYST_PROMPT": "src/multi_agent/analyst_pkg/prompt.py",
    "EXIT_ANALYST_PROMPT": "src/multi_agent/exit_analyst.py",
    "THESIS_GUARD_PROMPT": "src/multi_agent/thesis_guard.py",
    "ARBITER_PROMPT": "src/multi_agent/arbiter.py",
    "AUTONOMY_PROMPT": "src/brain/autonomy_prompt.py",
}


def _extract_prompt(file_path: Path, prompt_name: str) -> str | None:
    """Extract a prompt constant from a Python file."""
    if not file_path.exists():
        return None

    content = file_path.read_text()
    if prompt_name not in content:
        return None

    # Find triple-quoted string after the constant name
    start = content.find(prompt_name)
    quote_start = content.find('"""', start)
    if quote_start == -1:
        quote_start = content.find("'''", start)

    if quote_start > -1:
        quote_end = content.find('"""', quote_start + 3)
        if quote_end == -1:
            quote_end = content.find("'''", quote_start + 3)
        if quote_end > -1:
            return content[quote_start + 3 : quote_end].strip()

    return None


@mcp.tool()
def promptcompare(
    mode: str = "list",
    prompt_name: str | None = None,
    prompt_name_b: str | None = None,
    custom_prompt: str | None = None,
    focus: str = "trading",
) -> dict:
    """
    Analyze and compare trading prompts.

    Args:
        mode: Operation - list, analyze, compare, suggest
        prompt_name: Name of prompt to analyze (e.g., MARKET_SCANNER_PROMPT)
        prompt_name_b: Second prompt name for comparison
        custom_prompt: Custom prompt text to analyze
        focus: Analysis focus - trading, clarity, completeness

    Returns:
        Prompt analysis results
    """
    project_path = Path(SENTIENT_TRADER_PATH)

    if mode == "list":
        available = {}
        for name, path in PROMPT_LOCATIONS.items():
            full_path = project_path / path
            available[name] = {
                "path": path,
                "exists": full_path.exists(),
            }
        return {"mode": "list", "prompts": available}

    if mode == "analyze":
        if custom_prompt:
            prompt_text = custom_prompt
            name = "custom"
        elif prompt_name and prompt_name in PROMPT_LOCATIONS:
            prompt_text = _extract_prompt(project_path / PROMPT_LOCATIONS[prompt_name], prompt_name)
            name = prompt_name
        else:
            return {"error": f"Unknown prompt: {prompt_name}", "available": list(PROMPT_LOCATIONS.keys())}

        if not prompt_text:
            return {"error": f"Could not extract prompt: {prompt_name}"}

        # Analyze
        analysis = {
            "name": name,
            "metrics": {
                "words": len(prompt_text.split()),
                "lines": len(prompt_text.split("\n")),
                "chars": len(prompt_text),
            },
            "structure": {
                "has_role": "role" in prompt_text.lower() or "you are" in prompt_text.lower(),
                "has_rules": any(w in prompt_text.lower() for w in ["rule", "must", "never"]),
                "has_examples": "example" in prompt_text.lower(),
                "has_output_format": any(w in prompt_text.lower() for w in ["json", "format", "respond"]),
            },
        }

        if focus == "trading":
            analysis["trading"] = {
                "has_risk": any(w in prompt_text.lower() for w in ["risk", "stop", "loss"]),
                "has_entry": any(w in prompt_text.lower() for w in ["entry", "enter"]),
                "has_exit": any(w in prompt_text.lower() for w in ["exit", "close"]),
                "has_confidence": any(w in prompt_text.lower() for w in ["confidence", "conviction"]),
            }

        analysis["preview"] = prompt_text[:1000]
        return {"mode": "analyze", "analysis": analysis}

    if mode == "compare":
        if not prompt_name or not prompt_name_b:
            return {"error": "Provide both prompt_name and prompt_name_b"}

        prompt_a = _extract_prompt(project_path / PROMPT_LOCATIONS.get(prompt_name, ""), prompt_name)
        prompt_b = _extract_prompt(project_path / PROMPT_LOCATIONS.get(prompt_name_b, ""), prompt_name_b)

        if not prompt_a or not prompt_b:
            return {"error": "Could not extract one or both prompts"}

        words_a = set(prompt_a.lower().split())
        words_b = set(prompt_b.lower().split())

        return {
            "mode": "compare",
            "comparison": {
                "a": {"name": prompt_name, "words": len(prompt_a.split())},
                "b": {"name": prompt_name_b, "words": len(prompt_b.split())},
                "common_words": len(words_a & words_b),
                "unique_to_a": len(words_a - words_b),
                "unique_to_b": len(words_b - words_a),
            },
        }

    if mode == "suggest":
        prompt_text = custom_prompt
        if prompt_name and prompt_name in PROMPT_LOCATIONS:
            prompt_text = _extract_prompt(project_path / PROMPT_LOCATIONS[prompt_name], prompt_name)

        if not prompt_text:
            return {"error": "Provide prompt_name or custom_prompt"}

        suggestions = []
        if len(prompt_text) > 5000:
            suggestions.append("Consider breaking into smaller sections")
        if "you are" not in prompt_text.lower():
            suggestions.append("Add role definition (You are a...)")
        if "example" not in prompt_text.lower():
            suggestions.append("Add examples of expected behavior")
        if focus == "trading" and "risk" not in prompt_text.lower():
            suggestions.append("Add risk management guidance")

        return {"mode": "suggest", "suggestions": suggestions or ["Prompt looks well-structured"]}

    return {"error": f"Unknown mode: {mode}"}


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")
