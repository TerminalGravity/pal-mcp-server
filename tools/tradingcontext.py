"""
Trading Context tool - Fetch live trading system state for development context.

Bridges PAL MCP to the sentient-trader blofin-trading MCP server to provide
Claude Code with real-time context about:
- Current positions and P&L
- Recent trade history and performance
- Agent states and decisions
- Market conditions

This helps Claude Code make better decisions when developing trading logic.
"""

from __future__ import annotations

import logging
import os

import httpx
from mcp.types import TextContent
from pydantic import Field

from tools.shared.base_tool import BaseTool, BaseToolRequest

logger = logging.getLogger(__name__)

# Default endpoint for blofin-trading MCP server
BLOFIN_MCP_URL = os.getenv("BLOFIN_MCP_URL", "https://sentient-trader.fly.dev/mcp")


class TradingContextRequest(BaseToolRequest):
    """Request model for trading context tool."""

    context_type: str = Field(
        default="full",
        description=(
            "Type of context to fetch: "
            "'full' (positions + recent trades + market), "
            "'positions' (current positions only), "
            "'trades' (recent trade history), "
            "'market' (current market snapshot), "
            "'agents' (agent states and decisions), "
            "'performance' (session performance metrics)"
        ),
    )
    symbol: str = Field(
        default="BTC-USDT",
        description="Trading symbol to get context for",
    )
    trade_limit: int = Field(
        default=10,
        description="Number of recent trades to include",
    )


class TradingContextTool(BaseTool):
    """
    Fetches live trading system context for development assistance.

    This tool calls the blofin-trading MCP server to get real-time data
    that helps Claude Code understand the current state of the trading
    system when developing or debugging.
    """

    name = "tradingcontext"
    description = (
        "Get live trading system context (positions, trades, market, agents) "
        "to inform development decisions. Use before modifying trading logic."
    )
    request_model = TradingContextRequest

    async def execute(self, request: TradingContextRequest) -> list[TextContent]:
        """Execute the trading context fetch."""
        try:
            context_parts = []

            async with httpx.AsyncClient(timeout=30.0) as client:
                if request.context_type in ("full", "positions"):
                    positions = await self._fetch_positions(client, request.symbol)
                    context_parts.append(positions)

                if request.context_type in ("full", "trades"):
                    trades = await self._fetch_trades(client, request.symbol, request.trade_limit)
                    context_parts.append(trades)

                if request.context_type in ("full", "market"):
                    market = await self._fetch_market(client, request.symbol)
                    context_parts.append(market)

                if request.context_type in ("full", "agents"):
                    agents = await self._fetch_agents(client)
                    context_parts.append(agents)

                if request.context_type == "performance":
                    perf = await self._fetch_performance(client)
                    context_parts.append(perf)

            combined = "\n\n---\n\n".join(context_parts)
            return [TextContent(type="text", text=combined)]

        except Exception as e:
            logger.error(f"Trading context fetch failed: {e}")
            return [
                TextContent(
                    type="text",
                    text=f"⚠️ Failed to fetch trading context: {e}\n\n"
                    f"Ensure blofin-trading MCP server is running at {BLOFIN_MCP_URL}",
                )
            ]

    async def _call_mcp_tool(self, client: httpx.AsyncClient, tool: str, args: dict) -> dict:
        """Call a tool on the blofin-trading MCP server."""
        response = await client.post(
            f"{BLOFIN_MCP_URL}/tools/{tool}",
            json=args,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json()

    async def _fetch_positions(self, client: httpx.AsyncClient, symbol: str) -> str:
        """Fetch current positions."""
        try:
            result = await self._call_mcp_tool(client, "get_positions", {"symbol": symbol})
            positions = result.get("data", {}).get("positions", [])

            if not positions:
                return "## Current Positions\n\n**FLAT** - No open positions"

            lines = ["## Current Positions\n"]
            for pos in positions:
                lines.append(f"- **{pos.get('direction', 'unknown').upper()}** {symbol}")
                lines.append(f"  - Entry: ${pos.get('entry_price', 0):,.2f}")
                lines.append(f"  - Size: {pos.get('size', 0)} contracts")
                lines.append(f"  - Unrealized P&L: ${pos.get('unrealized_pnl', 0):,.2f}")
                lines.append(f"  - Leverage: {pos.get('leverage', 0)}x")

            return "\n".join(lines)
        except Exception as e:
            return f"## Current Positions\n\n⚠️ Error fetching positions: {e}"

    async def _fetch_trades(self, client: httpx.AsyncClient, symbol: str, limit: int) -> str:
        """Fetch recent trade history."""
        try:
            result = await self._call_mcp_tool(client, "get_db_trades", {"symbol": symbol, "limit": limit})
            trades = result.get("data", {}).get("trades", [])

            if not trades:
                return "## Recent Trades\n\nNo trades in history"

            lines = ["## Recent Trades\n"]
            wins = losses = total_pnl = 0

            for trade in trades:
                pnl = trade.get("realized_pnl", 0)
                direction = trade.get("direction", "unknown")
                entry = trade.get("entry_price", 0)
                exit_price = trade.get("exit_price", 0)
                emoji = "✅" if pnl > 0 else "❌" if pnl < 0 else "➖"

                lines.append(f"{emoji} **{direction.upper()}** @ ${entry:,.2f} → ${exit_price:,.2f} = ${pnl:+,.2f}")

                if pnl > 0:
                    wins += 1
                elif pnl < 0:
                    losses += 1
                total_pnl += pnl

            win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
            lines.append(f"\n**Summary:** {wins}W/{losses}L ({win_rate:.0f}% WR) | Net: ${total_pnl:+,.2f}")

            return "\n".join(lines)
        except Exception as e:
            return f"## Recent Trades\n\n⚠️ Error fetching trades: {e}"

    async def _fetch_market(self, client: httpx.AsyncClient, symbol: str) -> str:
        """Fetch current market snapshot."""
        try:
            result = await self._call_mcp_tool(client, "get_market_snapshot", {"symbol": symbol})
            data = result.get("data", {})

            lines = [
                "## Market Snapshot\n",
                f"- **Price:** ${data.get('price', 0):,.2f}",
                f"- **VWAP:** ${data.get('vwap', 0):,.2f} ({data.get('vwap_position', 'unknown')})",
                f"- **EMA Cloud:** {data.get('ema_cloud_state', 'unknown')}",
                f"- **Trend:** {data.get('trend', 'unknown')}",
                f"- **ATR:** ${data.get('atr', 0):,.2f}",
            ]

            return "\n".join(lines)
        except Exception as e:
            return f"## Market Snapshot\n\n⚠️ Error fetching market: {e}"

    async def _fetch_agents(self, client: httpx.AsyncClient) -> str:
        """Fetch agent states and recent decisions."""
        try:
            result = await self._call_mcp_tool(client, "get_agent_states", {})
            states = result.get("data", {}).get("agents", [])

            lines = ["## Agent States\n"]
            for agent in states:
                name = agent.get("name", "unknown")
                status = agent.get("status", "unknown")
                last_action = agent.get("last_action", "none")
                lines.append(f"- **{name}**: {status} (last: {last_action})")

            # Also get recent decisions
            decisions = await self._call_mcp_tool(client, "get_decisions", {"limit": 5})
            decision_list = decisions.get("data", {}).get("decisions", [])

            if decision_list:
                lines.append("\n### Recent Decisions\n")
                for dec in decision_list[:5]:
                    action = dec.get("action", "unknown")
                    confidence = dec.get("confidence", 0)
                    reasoning = dec.get("reasoning", "")[:100]
                    lines.append(f"- **{action}** ({confidence:.0%}): {reasoning}...")

            return "\n".join(lines)
        except Exception as e:
            return f"## Agent States\n\n⚠️ Error fetching agents: {e}"

    async def _fetch_performance(self, client: httpx.AsyncClient) -> str:
        """Fetch session performance metrics."""
        try:
            result = await self._call_mcp_tool(client, "get_session_stats", {})
            data = result.get("data", {})

            lines = [
                "## Session Performance\n",
                f"- **Trades:** {data.get('trades_executed', 0)}",
                f"- **Win Rate:** {data.get('win_rate', 0):.1%}",
                f"- **Net P&L:** ${data.get('net_pnl', 0):+,.2f}",
                f"- **Gross P&L:** ${data.get('gross_pnl', 0):+,.2f}",
                f"- **Fees:** ${data.get('total_fees', 0):,.2f}",
                f"- **Avg Trade:** ${data.get('avg_trade_pnl', 0):+,.2f}",
                f"- **Largest Win:** ${data.get('largest_win', 0):+,.2f}",
                f"- **Largest Loss:** ${data.get('largest_loss', 0):+,.2f}",
            ]

            return "\n".join(lines)
        except Exception as e:
            return f"## Session Performance\n\n⚠️ Error fetching performance: {e}"


# Export for tool registry
tool = TradingContextTool()
