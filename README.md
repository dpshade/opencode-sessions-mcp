# OpenCode Sessions MCP

An MCP server that exposes [opencode](https://opencode.ai) sessions for summarization, search, and analysis.

Built with [FastMCP](https://github.com/jlowin/fastmcp) based on [InteractionCo/mcp-server-template](https://github.com/InteractionCo/mcp-server-template).

## What It Does

Query your local opencode session history via MCP tools:
- "Summarize what I worked on today"
- "Find sessions about authentication"  
- "Show my token usage this week"

## Tools

| Tool | Description |
|------|-------------|
| `list_sessions` | List sessions with filtering (date, project, search) |
| `get_todays_sessions` | Get today's sessions with optional content for summarization |
| `get_session` | Full session data (messages, parts, tool calls) |
| `search_sessions` | FTS5 full-text search across all content |
| `get_session_stats` | Token usage, costs, model breakdown |
| `export_session` | Export as JSON, Markdown, or plain text |
| `list_projects` | All projects with session counts |
| `rebuild_search_index` | Refresh the SQLite cache |

## Date Filtering

The `list_sessions` tool accepts human-friendly date filters:

```python
list_sessions(date="today")           # Sessions from today
list_sessions(date="yesterday")       # Yesterday's sessions
list_sessions(date="7d")              # Last 7 days
list_sessions(date="30d")             # Last 30 days
list_sessions(date="this week")       # Since Monday
list_sessions(date="2026-01-06")      # Specific date
list_sessions(date="2026-01-01 to 2026-01-06")  # Date range
```

## How It Works

OpenCode stores sessions as JSON files at `~/.local/share/opencode/storage/`:

```
storage/
├── session/<project-id>/<session-id>.json   # Metadata, timestamps
├── message/<session-id>/<message-id>.json   # Turns, tokens, costs
└── part/<message-id>/<part-id>.json         # Text, tool calls, files
```

This MCP:
1. Scans all session files across all projects
2. Builds a SQLite cache with FTS5 indexes
3. Exposes tools for querying via MCP protocol

## Setup

### Local Development

```bash
git clone https://github.com/dps/opencode-sessions-mcp.git
cd opencode-sessions-mcp
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/server.py
```

### Test with MCP Inspector

```bash
npx @modelcontextprotocol/inspector
```

Connect to `http://localhost:8000/mcp` using "Streamable HTTP" transport.

### Deploy to Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

## Usage Examples

### Get Today's Sessions for Summary

```json
{
  "tool": "get_todays_sessions",
  "arguments": {
    "include_content": true
  }
}
```

### Search All Sessions

```json
{
  "tool": "search_sessions",
  "arguments": {
    "query": "authentication JWT",
    "search_content": true
  }
}
```

### Get Token Stats This Week

```json
{
  "tool": "get_session_stats",
  "arguments": {
    "date_from": 1736121600000
  }
}
```

### Export Session as Markdown

```json
{
  "tool": "export_session",
  "arguments": {
    "session_id": "ses_...",
    "format": "markdown"
  }
}
```

## Poke Integration

Connect at [poke.com/settings/connections](https://poke.com/settings/connections) then:
- "Summarize my coding sessions from today"
- "What did I work on yesterday?"
- "Find sessions where I debugged auth issues"

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |

## License

MIT
