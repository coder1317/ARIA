"""Tests for Browser and MCP modules — structure and import tests."""
import pytest


def test_browser_import():
    from ultra.tools.browser import Browser, BrowserResult, _HAS_PLAYWRIGHT
    assert hasattr(Browser, 'goto')
    assert hasattr(Browser, 'screenshot')
    assert hasattr(Browser, 'close')
    # BrowserResult should be a dataclass
    r = BrowserResult(url="http://test.com")
    assert r.success is True
    assert r.url == "http://test.com"
    assert "loaded" in r.summary


def test_browser_result_error():
    from ultra.tools.browser import BrowserResult
    r = BrowserResult(url="http://test.com", error="timeout", success=False)
    assert "timeout" in r.summary
    assert not r.success


def test_mcp_import():
    from ultra.tools.mcp_client import MCPManager, MCPTool, MCPServer, _HAS_MCP
    manager = MCPManager()
    assert manager.servers == {}
    assert manager.available_tools() == []


def test_mcp_tool_to_prompt():
    from ultra.tools.mcp_client import MCPTool
    t = MCPTool(
        name="read_file",
        description="Read a file from disk",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
            },
            "required": ["path"],
        },
        server_name="filesystem",
    )
    prompt = t.to_prompt()
    assert "read_file" in prompt
    assert "Read a file" in prompt
    assert "path" in prompt
    assert "required" in prompt


def test_mcp_add_server():
    from ultra.tools.mcp_client import MCPManager
    m = MCPManager()
    m.add_server("npx -y @modelcontextprotocol/server-filesystem /tmp", name="fs")
    assert "fs" in m.servers
    assert m.servers["fs"].command == "npx"
    assert m.servers["fs"].args == ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]


def test_channel_import():
    from ultra.channels import ChannelAdapter, NormalizedMessage, NormalizedResponse
    msg = NormalizedMessage(channel="cli", user_id="test", text="hello")
    assert msg.channel == "cli"
    assert not msg.is_group

    resp = NormalizedResponse(text="hi there")
    assert resp.text == "hi there"
    assert resp.parse_mode == "markdown"
