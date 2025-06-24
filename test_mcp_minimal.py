#!/usr/bin/env python3
"""
Minimal test to reproduce the MCP server startup issue.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)

logger = logging.getLogger(__name__)

async def test_server_creation():
    """Test server creation without stdio transport."""
    try:
        logger.info("Testing server creation...")
        from graph_of_thoughts.mcp_server import create_server
        
        server = await create_server()
        logger.info("✅ Server created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Server creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_stdio_import():
    """Test stdio transport import."""
    try:
        logger.info("Testing stdio transport import...")
        from mcp.server.stdio import stdio_server
        logger.info("✅ stdio_server imported successfully")
        return True
    except Exception as e:
        logger.error(f"❌ stdio_server import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_stdio_context():
    """Test stdio transport context manager."""
    try:
        logger.info("Testing stdio transport context...")
        from mcp.server.stdio import stdio_server
        
        # Test if we can create the context (but don't enter it)
        context = stdio_server()
        logger.info("✅ stdio_server context created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ stdio_server context failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_full_server_setup():
    """Test full server setup without running."""
    try:
        logger.info("Testing full server setup...")
        from graph_of_thoughts.mcp_server import create_server
        from mcp.server.stdio import stdio_server
        
        # Create server
        got_server = await create_server()
        logger.info("✅ Server created")
        
        # Test stdio context creation
        context = stdio_server()
        logger.info("✅ stdio context created")
        
        logger.info("✅ Full server setup successful (not running)")
        return True
    except Exception as e:
        logger.error(f"❌ Full server setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    logger.info("🔧 Starting MCP Server Diagnostic Tests")
    
    tests = [
        ("Server Creation", test_server_creation),
        ("Stdio Import", test_stdio_import),
        ("Stdio Context", test_stdio_context),
        ("Full Setup", test_full_server_setup),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        result = await test_func()
        results.append((test_name, result))
    
    logger.info("\n📊 Test Results:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        logger.info("\n🎉 All tests passed! The issue might be with the MCP client connection.")
    else:
        logger.info("\n⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    asyncio.run(main())
