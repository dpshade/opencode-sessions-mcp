#!/usr/bin/env python3
"""
OpenCode Sessions MCP Server

Exposes opencode sessions for summarization, search, and analysis.
Storage location: ~/.local/share/opencode/storage/
"""

import json
import os
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator
from contextlib import contextmanager

from fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("OpenCode Sessions")

# Storage paths
STORAGE_DIR = Path.home() / ".local/share/opencode/storage"
CACHE_DB = Path.home() / ".local/share/opencode/sessions_cache.db"


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class SessionInfo:
    id: str
    project_id: str
    directory: str
    title: str
    version: str
    created: int  # Unix timestamp ms
    updated: int  # Unix timestamp ms
    parent_id: Optional[str] = None
    additions: int = 0
    deletions: int = 0
    files_changed: int = 0


@dataclass
class MessageInfo:
    id: str
    session_id: str
    role: str  # "user" | "assistant"
    created: int
    agent: str
    model_id: str
    provider_id: str
    cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    completed: Optional[int] = None


@dataclass
class PartInfo:
    id: str
    message_id: str
    session_id: str
    type: str  # "text" | "tool" | "file" | etc.
    content: str  # Text content or JSON-serialized data


# =============================================================================
# Storage Access Layer
# =============================================================================


def get_storage_dir() -> Path:
    """Get the opencode storage directory."""
    return STORAGE_DIR


def discover_all_projects() -> List[str]:
    """Discover all project IDs (including 'global')."""
    session_dir = get_storage_dir() / "session"
    if not session_dir.exists():
        return []
    return [d.name for d in session_dir.iterdir() if d.is_dir()]


def load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    """Safely load a JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, PermissionError):
        return None


def iter_sessions(project_id: Optional[str] = None) -> Iterator[SessionInfo]:
    """Iterate over all sessions, optionally filtered by project."""
    session_base = get_storage_dir() / "session"
    if not session_base.exists():
        return

    projects = [project_id] if project_id else discover_all_projects()

    for proj in projects:
        proj_dir = session_base / proj
        if not proj_dir.exists():
            continue
        for session_file in proj_dir.glob("*.json"):
            data = load_json_file(session_file)
            if data:
                summary = data.get("summary", {})
                yield SessionInfo(
                    id=data.get("id", ""),
                    project_id=data.get("projectID", proj),
                    directory=data.get("directory", ""),
                    title=data.get("title", "Untitled"),
                    version=data.get("version", ""),
                    created=data.get("time", {}).get("created", 0),
                    updated=data.get("time", {}).get("updated", 0),
                    parent_id=data.get("parentID"),
                    additions=summary.get("additions", 0),
                    deletions=summary.get("deletions", 0),
                    files_changed=summary.get("files", 0),
                )


def iter_messages(session_id: str) -> Iterator[MessageInfo]:
    """Iterate over all messages for a session."""
    message_dir = get_storage_dir() / "message" / session_id
    if not message_dir.exists():
        return

    for msg_file in message_dir.glob("*.json"):
        data = load_json_file(msg_file)
        if data:
            tokens = data.get("tokens", {})
            cache = tokens.get("cache", {})
            model = data.get("model", {})
            yield MessageInfo(
                id=data.get("id", ""),
                session_id=data.get("sessionID", session_id),
                role=data.get("role", ""),
                created=data.get("time", {}).get("created", 0),
                agent=data.get("agent", ""),
                model_id=model.get("modelID", data.get("modelID", "")),
                provider_id=model.get("providerID", data.get("providerID", "")),
                cost=data.get("cost", 0.0),
                input_tokens=tokens.get("input", 0),
                output_tokens=tokens.get("output", 0),
                reasoning_tokens=tokens.get("reasoning", 0),
                cache_read=cache.get("read", 0),
                cache_write=cache.get("write", 0),
                completed=data.get("time", {}).get("completed"),
            )


def iter_parts(message_id: str, session_id: str) -> Iterator[PartInfo]:
    """Iterate over all parts for a message."""
    part_dir = get_storage_dir() / "part" / message_id
    if not part_dir.exists():
        return

    for part_file in part_dir.glob("*.json"):
        data = load_json_file(part_file)
        if data:
            part_type = data.get("type", "unknown")
            # Extract text content based on type
            if part_type == "text":
                content = data.get("text", "")
            elif part_type == "tool":
                # Include tool name and input for searchability
                tool_name = data.get("tool", "")
                state = data.get("state", {})
                tool_input = state.get("input", {})
                output = state.get("output", "")
                content = json.dumps(
                    {
                        "tool": tool_name,
                        "input": tool_input,
                        "output": output[:5000] if isinstance(output, str) else output,
                    }
                )
            else:
                content = json.dumps(data)

            yield PartInfo(
                id=data.get("id", ""),
                message_id=data.get("messageID", message_id),
                session_id=data.get("sessionID", session_id),
                type=part_type,
                content=content,
            )


# =============================================================================
# SQLite Cache with FTS5
# =============================================================================


@contextmanager
def get_db():
    """Get database connection with FTS5 support."""
    conn = sqlite3.connect(str(CACHE_DB))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Initialize the SQLite database with FTS5 tables."""
    with get_db() as conn:
        conn.executescript("""
            -- Sessions table
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                directory TEXT,
                title TEXT,
                version TEXT,
                created INTEGER,
                updated INTEGER,
                parent_id TEXT,
                additions INTEGER DEFAULT 0,
                deletions INTEGER DEFAULT 0,
                files_changed INTEGER DEFAULT 0
            );
            
            CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created);
            CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated);
            CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id);
            
            -- Messages table
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                created INTEGER,
                agent TEXT,
                model_id TEXT,
                provider_id TEXT,
                cost REAL DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                reasoning_tokens INTEGER DEFAULT 0,
                cache_read INTEGER DEFAULT 0,
                cache_write INTEGER DEFAULT 0,
                completed INTEGER,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            
            -- Parts table
            CREATE TABLE IF NOT EXISTS parts (
                id TEXT PRIMARY KEY,
                message_id TEXT,
                session_id TEXT,
                type TEXT,
                content TEXT,
                FOREIGN KEY (message_id) REFERENCES messages(id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_parts_message ON parts(message_id);
            CREATE INDEX IF NOT EXISTS idx_parts_session ON parts(session_id);
            
            -- FTS5 for full-text search
            CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
                id,
                title,
                directory,
                content='sessions',
                content_rowid='rowid'
            );
            
            CREATE VIRTUAL TABLE IF NOT EXISTS parts_fts USING fts5(
                id,
                session_id,
                content,
                content='parts',
                content_rowid='rowid'
            );
            
            -- Metadata table for tracking last index time
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        conn.commit()


def rebuild_index():
    """Rebuild the entire search index from storage files."""
    init_db()

    with get_db() as conn:
        # Clear existing data
        conn.execute("DELETE FROM parts")
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM sessions")
        conn.execute("DELETE FROM sessions_fts")
        conn.execute("DELETE FROM parts_fts")

        session_count = 0
        message_count = 0
        part_count = 0

        # Index all sessions
        for session in iter_sessions():
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions 
                (id, project_id, directory, title, version, created, updated, 
                 parent_id, additions, deletions, files_changed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session.id,
                    session.project_id,
                    session.directory,
                    session.title,
                    session.version,
                    session.created,
                    session.updated,
                    session.parent_id,
                    session.additions,
                    session.deletions,
                    session.files_changed,
                ),
            )
            session_count += 1

            # Index messages for this session
            for message in iter_messages(session.id):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO messages
                    (id, session_id, role, created, agent, model_id, provider_id,
                     cost, input_tokens, output_tokens, reasoning_tokens,
                     cache_read, cache_write, completed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        message.id,
                        message.session_id,
                        message.role,
                        message.created,
                        message.agent,
                        message.model_id,
                        message.provider_id,
                        message.cost,
                        message.input_tokens,
                        message.output_tokens,
                        message.reasoning_tokens,
                        message.cache_read,
                        message.cache_write,
                        message.completed,
                    ),
                )
                message_count += 1

                # Index parts for this message
                for part in iter_parts(message.id, session.id):
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO parts
                        (id, message_id, session_id, type, content)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (
                            part.id,
                            part.message_id,
                            part.session_id,
                            part.type,
                            part.content,
                        ),
                    )
                    part_count += 1

        # Rebuild FTS indexes
        conn.execute("""
            INSERT INTO sessions_fts(sessions_fts) VALUES('rebuild')
        """)
        conn.execute("""
            INSERT INTO parts_fts(parts_fts) VALUES('rebuild')
        """)

        # Update metadata
        conn.execute(
            """
            INSERT OR REPLACE INTO metadata (key, value) 
            VALUES ('last_indexed', ?)
        """,
            (datetime.now().isoformat(),),
        )

        conn.commit()

        return {
            "sessions": session_count,
            "messages": message_count,
            "parts": part_count,
            "indexed_at": datetime.now().isoformat(),
        }


# =============================================================================
# MCP Tools
# =============================================================================


def parse_date_filter(date_str: str) -> tuple[int, int]:
    """
    Parse a date string into (start_ms, end_ms) timestamps.

    Supports:
        - "today" / "yesterday" / "this week" / "last 7 days"
        - "YYYY-MM-DD" (specific date)
        - "YYYY-MM-DD to YYYY-MM-DD" (date range)
    """
    from datetime import timedelta

    date_str = date_str.lower().strip()
    now = datetime.now()
    today_start = datetime(now.year, now.month, now.day)

    if date_str == "today":
        start = today_start
        end = today_start + timedelta(days=1)
    elif date_str == "yesterday":
        start = today_start - timedelta(days=1)
        end = today_start
    elif date_str in ("this week", "week"):
        # Monday of this week
        start = today_start - timedelta(days=now.weekday())
        end = now
    elif date_str in ("last 7 days", "7 days", "7d"):
        start = today_start - timedelta(days=7)
        end = now
    elif date_str in ("last 30 days", "30 days", "30d", "month"):
        start = today_start - timedelta(days=30)
        end = now
    elif " to " in date_str:
        # Range: "2026-01-01 to 2026-01-06"
        parts = date_str.split(" to ")
        start = datetime.strptime(parts[0].strip(), "%Y-%m-%d")
        end = datetime.strptime(parts[1].strip(), "%Y-%m-%d") + timedelta(days=1)
    else:
        # Single date: "2026-01-06"
        try:
            start = datetime.strptime(date_str, "%Y-%m-%d")
            end = start + timedelta(days=1)
        except ValueError:
            raise ValueError(
                f"Unknown date format: {date_str}. Use 'today', 'yesterday', '7d', 'YYYY-MM-DD', or 'YYYY-MM-DD to YYYY-MM-DD'"
            )

    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


@mcp.tool(
    description="List all opencode sessions with optional filtering by project, date, and search query"
)
def list_sessions(
    project_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    sort_by: str = "updated",  # "updated" | "created"
    sort_order: str = "desc",  # "asc" | "desc"
    date: Optional[
        str
    ] = None,  # "today", "yesterday", "7d", "2026-01-06", "2026-01-01 to 2026-01-06"
    date_from: Optional[int] = None,  # Unix timestamp ms (legacy)
    date_to: Optional[int] = None,  # Unix timestamp ms (legacy)
    search: Optional[str] = None,  # Search in title
) -> Dict[str, Any]:
    """
    List sessions with filtering and pagination.

    Args:
        project_id: Filter by project (git commit SHA or 'global')
        limit: Max sessions to return (default 50)
        offset: Pagination offset
        sort_by: Sort field ('updated' or 'created')
        sort_order: Sort direction ('asc' or 'desc')
        date: Human-friendly date filter: 'today', 'yesterday', '7d', '30d', 'this week',
              'YYYY-MM-DD', or 'YYYY-MM-DD to YYYY-MM-DD'
        date_from: Filter sessions after this timestamp (ms) - use 'date' instead
        date_to: Filter sessions before this timestamp (ms) - use 'date' instead
        search: Search query for session titles

    Returns:
        Dict with sessions list, total count, and pagination info
    """
    init_db()

    # Parse human-friendly date filter
    if date:
        try:
            date_from, date_to = parse_date_filter(date)
        except ValueError as e:
            return {"error": str(e)}

    with get_db() as conn:
        # Build query
        conditions = []
        params = []

        if project_id:
            conditions.append("project_id = ?")
            params.append(project_id)

        if date_from:
            conditions.append(f"{sort_by} >= ?")
            params.append(date_from)

        if date_to:
            conditions.append(f"{sort_by} <= ?")
            params.append(date_to)

        if search:
            # Use FTS for search
            conditions.append(
                "id IN (SELECT id FROM sessions_fts WHERE sessions_fts MATCH ?)"
            )
            params.append(search)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        order = "DESC" if sort_order.lower() == "desc" else "ASC"

        # Get total count
        count_query = f"SELECT COUNT(*) FROM sessions WHERE {where_clause}"
        total = conn.execute(count_query, params).fetchone()[0]

        # Get sessions
        query = f"""
            SELECT * FROM sessions 
            WHERE {where_clause}
            ORDER BY {sort_by} {order}
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()

        sessions = []
        for row in rows:
            sessions.append(
                {
                    "id": row["id"],
                    "project_id": row["project_id"],
                    "directory": row["directory"],
                    "title": row["title"],
                    "version": row["version"],
                    "created": row["created"],
                    "created_human": datetime.fromtimestamp(
                        row["created"] / 1000
                    ).isoformat()
                    if row["created"]
                    else None,
                    "updated": row["updated"],
                    "updated_human": datetime.fromtimestamp(
                        row["updated"] / 1000
                    ).isoformat()
                    if row["updated"]
                    else None,
                    "parent_id": row["parent_id"],
                    "summary": {
                        "additions": row["additions"],
                        "deletions": row["deletions"],
                        "files": row["files_changed"],
                    },
                }
            )

        return {
            "sessions": sessions,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total,
        }


@mcp.tool(description="Get all sessions from today - perfect for daily summaries")
def get_todays_sessions(
    include_content: bool = False,
    max_content_length: int = 5000,
) -> Dict[str, Any]:
    """
    Get all sessions updated today. Convenience method for daily summaries.

    Args:
        include_content: If True, includes text content from each session (for summarization)
        max_content_length: Max chars of content per session when include_content=True

    Returns:
        List of today's sessions with optional content for summarization
    """
    # Get today's sessions
    result = list_sessions(
        date="today", limit=100, sort_by="updated", sort_order="desc"
    )

    if "error" in result:
        return result

    sessions = result.get("sessions", [])

    if include_content:
        for session in sessions:
            # Get condensed content for summarization
            session_data = get_session(
                session["id"], include_parts=True, max_part_length=max_content_length
            )

            # Extract just the text parts for summarization
            content_parts = []
            for msg in session_data.get("messages", []):
                role = msg["role"]
                for part in msg.get("parts", []):
                    if part["type"] == "text" and part["content"].strip():
                        content_parts.append(f"[{role}]: {part['content'][:1000]}")

            session["content_preview"] = "\n".join(
                content_parts[:20]
            )  # First 20 exchanges
            session["message_count"] = len(session_data.get("messages", []))

    return {
        "date": "today",
        "sessions": sessions,
        "total": len(sessions),
    }


@mcp.tool(
    description="Get full session details including all messages and their content parts"
)
def get_session(
    session_id: str,
    include_parts: bool = True,
    max_part_length: int = 10000,
) -> Dict[str, Any]:
    """
    Get complete session data for summarization.

    Args:
        session_id: The session ID (e.g., 'ses_...')
        include_parts: Whether to include message parts (default True)
        max_part_length: Truncate part content beyond this length

    Returns:
        Complete session with messages and parts
    """
    init_db()

    with get_db() as conn:
        # Get session
        session_row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()

        if not session_row:
            return {"error": f"Session not found: {session_id}"}

        session = {
            "id": session_row["id"],
            "project_id": session_row["project_id"],
            "directory": session_row["directory"],
            "title": session_row["title"],
            "version": session_row["version"],
            "created": session_row["created"],
            "created_human": datetime.fromtimestamp(
                session_row["created"] / 1000
            ).isoformat()
            if session_row["created"]
            else None,
            "updated": session_row["updated"],
            "updated_human": datetime.fromtimestamp(
                session_row["updated"] / 1000
            ).isoformat()
            if session_row["updated"]
            else None,
            "parent_id": session_row["parent_id"],
            "summary": {
                "additions": session_row["additions"],
                "deletions": session_row["deletions"],
                "files": session_row["files_changed"],
            },
        }

        # Get messages ordered by creation time
        message_rows = conn.execute(
            """
            SELECT * FROM messages 
            WHERE session_id = ? 
            ORDER BY created ASC
        """,
            (session_id,),
        ).fetchall()

        messages = []
        for msg_row in message_rows:
            message = {
                "id": msg_row["id"],
                "role": msg_row["role"],
                "created": msg_row["created"],
                "created_human": datetime.fromtimestamp(
                    msg_row["created"] / 1000
                ).isoformat()
                if msg_row["created"]
                else None,
                "agent": msg_row["agent"],
                "model": {
                    "model_id": msg_row["model_id"],
                    "provider_id": msg_row["provider_id"],
                },
                "cost": msg_row["cost"],
                "tokens": {
                    "input": msg_row["input_tokens"],
                    "output": msg_row["output_tokens"],
                    "reasoning": msg_row["reasoning_tokens"],
                    "cache_read": msg_row["cache_read"],
                    "cache_write": msg_row["cache_write"],
                },
            }

            if include_parts:
                part_rows = conn.execute(
                    """
                    SELECT * FROM parts 
                    WHERE message_id = ? 
                    ORDER BY id ASC
                """,
                    (msg_row["id"],),
                ).fetchall()

                parts = []
                for part_row in part_rows:
                    content = part_row["content"]
                    if len(content) > max_part_length:
                        content = (
                            content[:max_part_length]
                            + f"... [truncated, {len(part_row['content'])} total chars]"
                        )

                    parts.append(
                        {
                            "id": part_row["id"],
                            "type": part_row["type"],
                            "content": content,
                        }
                    )

                message["parts"] = parts

            messages.append(message)

        session["messages"] = messages
        session["message_count"] = len(messages)

        return session


@mcp.tool(description="Search across all session content using full-text search")
def search_sessions(
    query: str,
    limit: int = 20,
    search_titles: bool = True,
    search_content: bool = True,
) -> Dict[str, Any]:
    """
    Full-text search across sessions and message content.

    Args:
        query: Search query (supports FTS5 syntax: AND, OR, NOT, "phrases")
        limit: Max results to return
        search_titles: Search in session titles
        search_content: Search in message/part content

    Returns:
        Matching sessions with relevance info
    """
    init_db()

    results = []

    with get_db() as conn:
        if search_titles:
            # Search session titles
            title_matches = conn.execute(
                """
                SELECT s.*, 
                       highlight(sessions_fts, 1, '<mark>', '</mark>') as title_highlight
                FROM sessions_fts 
                JOIN sessions s ON sessions_fts.id = s.id
                WHERE sessions_fts MATCH ?
                LIMIT ?
            """,
                (query, limit),
            ).fetchall()

            for row in title_matches:
                results.append(
                    {
                        "session_id": row["id"],
                        "title": row["title"],
                        "title_highlight": row["title_highlight"],
                        "directory": row["directory"],
                        "updated": row["updated"],
                        "updated_human": datetime.fromtimestamp(
                            row["updated"] / 1000
                        ).isoformat()
                        if row["updated"]
                        else None,
                        "match_type": "title",
                    }
                )

        if search_content:
            # Search part content
            content_matches = conn.execute(
                """
                SELECT p.session_id, p.content, p.type, s.title, s.directory, s.updated,
                       snippet(parts_fts, 2, '<mark>', '</mark>', '...', 32) as snippet
                FROM parts_fts
                JOIN parts p ON parts_fts.id = p.id
                JOIN sessions s ON p.session_id = s.id
                WHERE parts_fts MATCH ?
                LIMIT ?
            """,
                (query, limit),
            ).fetchall()

            # Group by session
            session_matches = {}
            for row in content_matches:
                sid = row["session_id"]
                if sid not in session_matches:
                    session_matches[sid] = {
                        "session_id": sid,
                        "title": row["title"],
                        "directory": row["directory"],
                        "updated": row["updated"],
                        "updated_human": datetime.fromtimestamp(
                            row["updated"] / 1000
                        ).isoformat()
                        if row["updated"]
                        else None,
                        "match_type": "content",
                        "snippets": [],
                    }
                session_matches[sid]["snippets"].append(
                    {
                        "type": row["type"],
                        "snippet": row["snippet"],
                    }
                )

            results.extend(session_matches.values())

    # Deduplicate by session_id
    seen = set()
    unique_results = []
    for r in results:
        if r["session_id"] not in seen:
            seen.add(r["session_id"])
            unique_results.append(r)

    return {
        "query": query,
        "results": unique_results[:limit],
        "total": len(unique_results),
    }


@mcp.tool(
    description="Get aggregated statistics for sessions (tokens, costs, tool usage)"
)
def get_session_stats(
    session_id: Optional[str] = None,
    project_id: Optional[str] = None,
    date_from: Optional[int] = None,
    date_to: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Get aggregated statistics across sessions.

    Args:
        session_id: Stats for specific session
        project_id: Stats for specific project
        date_from: Filter by date range start (ms)
        date_to: Filter by date range end (ms)

    Returns:
        Aggregated token usage, costs, and tool call frequency
    """
    init_db()

    with get_db() as conn:
        # Build conditions
        conditions = []
        params = []

        if session_id:
            conditions.append("m.session_id = ?")
            params.append(session_id)

        if project_id:
            conditions.append("s.project_id = ?")
            params.append(project_id)

        if date_from:
            conditions.append("s.updated >= ?")
            params.append(date_from)

        if date_to:
            conditions.append("s.updated <= ?")
            params.append(date_to)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Aggregate message stats
        stats_query = f"""
            SELECT 
                COUNT(DISTINCT s.id) as session_count,
                COUNT(m.id) as message_count,
                SUM(m.cost) as total_cost,
                SUM(m.input_tokens) as total_input_tokens,
                SUM(m.output_tokens) as total_output_tokens,
                SUM(m.reasoning_tokens) as total_reasoning_tokens,
                SUM(m.cache_read) as total_cache_read,
                SUM(m.cache_write) as total_cache_write
            FROM sessions s
            LEFT JOIN messages m ON s.id = m.session_id
            WHERE {where_clause}
        """

        row = conn.execute(stats_query, params).fetchone()

        # Get model usage breakdown
        model_query = f"""
            SELECT 
                m.provider_id,
                m.model_id,
                COUNT(*) as message_count,
                SUM(m.cost) as cost,
                SUM(m.input_tokens + m.output_tokens) as tokens
            FROM messages m
            JOIN sessions s ON m.session_id = s.id
            WHERE {where_clause} AND m.role = 'assistant'
            GROUP BY m.provider_id, m.model_id
            ORDER BY tokens DESC
        """

        model_rows = conn.execute(model_query, params).fetchall()

        # Get tool usage
        tool_query = f"""
            SELECT 
                p.type,
                COUNT(*) as count
            FROM parts p
            JOIN sessions s ON p.session_id = s.id
            WHERE {where_clause}
            GROUP BY p.type
            ORDER BY count DESC
        """

        tool_rows = conn.execute(tool_query, params).fetchall()

        return {
            "summary": {
                "session_count": row["session_count"] or 0,
                "message_count": row["message_count"] or 0,
                "total_cost_usd": round(row["total_cost"] or 0, 4),
                "tokens": {
                    "input": row["total_input_tokens"] or 0,
                    "output": row["total_output_tokens"] or 0,
                    "reasoning": row["total_reasoning_tokens"] or 0,
                    "cache_read": row["total_cache_read"] or 0,
                    "cache_write": row["total_cache_write"] or 0,
                    "total": (row["total_input_tokens"] or 0)
                    + (row["total_output_tokens"] or 0),
                },
            },
            "models": [
                {
                    "provider": r["provider_id"],
                    "model": r["model_id"],
                    "messages": r["message_count"],
                    "cost": round(r["cost"] or 0, 4),
                    "tokens": r["tokens"] or 0,
                }
                for r in model_rows
            ],
            "part_types": [{"type": r["type"], "count": r["count"]} for r in tool_rows],
        }


@mcp.tool(
    description="Export a session in different formats (JSON, markdown, or plain text)"
)
def export_session(
    session_id: str,
    format: str = "markdown",  # "json" | "markdown" | "text"
) -> str:
    """
    Export session in various formats for processing.

    Args:
        session_id: Session to export
        format: Output format ('json', 'markdown', 'text')

    Returns:
        Formatted session content
    """
    session_data = get_session(session_id, include_parts=True, max_part_length=50000)

    if "error" in session_data:
        return json.dumps(session_data)

    if format == "json":
        return json.dumps(session_data, indent=2)

    elif format == "markdown":
        lines = [
            f"# {session_data['title']}",
            "",
            f"**Session ID:** {session_data['id']}",
            f"**Directory:** {session_data['directory']}",
            f"**Created:** {session_data['created_human']}",
            f"**Updated:** {session_data['updated_human']}",
            "",
            "---",
            "",
        ]

        for msg in session_data.get("messages", []):
            role = msg["role"].upper()
            agent = msg.get("agent", "")
            model = msg.get("model", {}).get("model_id", "")

            lines.append(f"## {role} ({agent} - {model})")
            lines.append(f"*{msg.get('created_human', '')}*")
            lines.append("")

            for part in msg.get("parts", []):
                if part["type"] == "text":
                    lines.append(part["content"])
                elif part["type"] == "tool":
                    try:
                        tool_data = json.loads(part["content"])
                        lines.append(f"**Tool:** `{tool_data.get('tool', 'unknown')}`")
                        lines.append("```json")
                        lines.append(
                            json.dumps(tool_data.get("input", {}), indent=2)[:2000]
                        )
                        lines.append("```")
                    except json.JSONDecodeError:
                        lines.append(f"**Tool call:** {part['content'][:500]}")
                else:
                    lines.append(f"*[{part['type']}]*")
                lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    else:  # text
        lines = [
            f"Session: {session_data['title']}",
            f"ID: {session_data['id']}",
            f"Directory: {session_data['directory']}",
            f"Created: {session_data['created_human']}",
            "",
            "=" * 60,
            "",
        ]

        for msg in session_data.get("messages", []):
            role = msg["role"].upper()
            lines.append(f"[{role}] ({msg.get('agent', '')})")

            for part in msg.get("parts", []):
                if part["type"] == "text":
                    lines.append(part["content"])
                elif part["type"] == "tool":
                    try:
                        tool_data = json.loads(part["content"])
                        lines.append(f">>> Tool: {tool_data.get('tool', 'unknown')}")
                    except json.JSONDecodeError:
                        lines.append(f">>> Tool call")

            lines.append("")
            lines.append("-" * 40)
            lines.append("")

        return "\n".join(lines)


@mcp.tool(description="Rebuild the session search index from storage files")
def rebuild_search_index() -> Dict[str, Any]:
    """
    Rebuild the SQLite search index from opencode storage files.

    This scans all sessions, messages, and parts and updates the
    FTS5 search index. Run this after new sessions are created or
    if search results seem stale.

    Returns:
        Index statistics (session/message/part counts)
    """
    return rebuild_index()


@mcp.tool(description="Get index status and metadata")
def get_index_status() -> Dict[str, Any]:
    """
    Check the current state of the search index.

    Returns:
        Index metadata including last indexed time and counts
    """
    init_db()

    with get_db() as conn:
        # Get counts
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        message_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        part_count = conn.execute("SELECT COUNT(*) FROM parts").fetchone()[0]

        # Get last indexed time
        last_indexed = conn.execute(
            "SELECT value FROM metadata WHERE key = 'last_indexed'"
        ).fetchone()

        # Count files on disk
        disk_sessions = sum(
            len(list((get_storage_dir() / "session" / p).glob("*.json")))
            for p in discover_all_projects()
        )

        return {
            "indexed": {
                "sessions": session_count,
                "messages": message_count,
                "parts": part_count,
            },
            "on_disk": {
                "sessions": disk_sessions,
                "projects": discover_all_projects(),
            },
            "last_indexed": last_indexed[0] if last_indexed else None,
            "storage_path": str(STORAGE_DIR),
            "cache_db_path": str(CACHE_DB),
            "needs_reindex": disk_sessions != session_count,
        }


@mcp.tool(description="List all discovered projects (git repos and global)")
def list_projects() -> Dict[str, Any]:
    """
    List all projects that have opencode sessions.

    Returns:
        List of project IDs with session counts
    """
    init_db()

    with get_db() as conn:
        rows = conn.execute("""
            SELECT 
                project_id,
                COUNT(*) as session_count,
                MAX(updated) as last_updated,
                directory
            FROM sessions
            GROUP BY project_id
            ORDER BY last_updated DESC
        """).fetchall()

        return {
            "projects": [
                {
                    "id": row["project_id"],
                    "session_count": row["session_count"],
                    "last_updated": row["last_updated"],
                    "last_updated_human": datetime.fromtimestamp(
                        row["last_updated"] / 1000
                    ).isoformat()
                    if row["last_updated"]
                    else None,
                    "sample_directory": row["directory"],
                }
                for row in rows
            ],
            "total_projects": len(rows),
        }


# =============================================================================
# Server Entry Point
# =============================================================================

if __name__ == "__main__":
    # Initialize database on startup
    init_db()

    # Check if index is empty and auto-rebuild
    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        if count == 0:
            print("Index empty, rebuilding from storage...")
            result = rebuild_index()
            print(
                f"Indexed {result['sessions']} sessions, {result['messages']} messages, {result['parts']} parts"
            )

    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"

    print(f"Starting OpenCode Sessions MCP on {host}:{port}")
    print(f"Storage: {STORAGE_DIR}")
    print(f"Cache: {CACHE_DB}")

    mcp.run(transport="http", host=host, port=port, stateless_http=True)
