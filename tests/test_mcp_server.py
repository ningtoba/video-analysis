"""Tests for the MCP Tool Server module (video_analysis/mcp_server.py).

These tests verify the module structure and tool signatures without
actually running the MCP server or importing models.  The FastMCP
object is created at import time but tools are lazily registered.
"""

# Just test that the module imports cleanly and has the right structure
from video_analysis import mcp_server


class TestMcpServerModule:
    def test_module_has_mcp_object(self):
        """FastMCP server object is created at module level."""
        assert mcp_server.mcp is not None
        assert mcp_server.mcp.name == "video-analysis"

    def test_module_has_main(self):
        """main() entry point exists."""
        assert callable(mcp_server.main)

    def test_module_has_ensure_services(self):
        """_ensure_services function exists."""
        assert callable(mcp_server._ensure_services)

    def test_module_version(self):
        """mcp is initialised as a FastMCP object."""
        assert "FastMCP" in type(mcp_server.mcp).__name__

    def test_all_tools_are_coros(self):
        """All tool functions are async."""
        from inspect import iscoroutinefunction

        for tool_name in [
            "process_video",
            "search_videos",
            "ask_question",
            "extract_scenes",
            "detect_objects",
            "list_library",
            "delete_video",
        ]:
            fn = getattr(mcp_server, tool_name, None)
            assert fn is not None, f"Tool {tool_name} not found in module"
            assert iscoroutinefunction(fn), f"{tool_name} is not async"

    def test_process_video_params(self):
        import inspect

        sig = inspect.signature(mcp_server.process_video)
        params = list(sig.parameters.keys())
        assert "video" in params
        assert "url" in params
        assert "processing_mode" in params

    def test_search_videos_params(self):
        import inspect

        sig = inspect.signature(mcp_server.search_videos)
        params = list(sig.parameters.keys())
        assert "query" in params
        assert "top_k" in params

    def test_ask_question_params(self):
        import inspect

        sig = inspect.signature(mcp_server.ask_question)
        params = list(sig.parameters.keys())
        assert "question" in params
        assert "video_id" in params

    def test_extract_scenes_params(self):
        import inspect

        sig = inspect.signature(mcp_server.extract_scenes)
        assert "video_id" in sig.parameters

    def test_detect_objects_params(self):
        import inspect

        sig = inspect.signature(mcp_server.detect_objects)
        assert "video_path" in sig.parameters

    def test_list_library_no_params(self):
        import inspect

        sig = inspect.signature(mcp_server.list_library)
        # May have self for FastMCP internals
        assert len(sig.parameters) <= 1

    def test_delete_video_params(self):
        import inspect

        sig = inspect.signature(mcp_server.delete_video)
        assert "video_id" in sig.parameters
