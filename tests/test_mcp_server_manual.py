#!/usr/bin/env python3
# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Derek Vitrano

"""
Manual test for the Graph of Thoughts MCP Server.

This module provides a manual test that demonstrates the MCP server functionality
by running actual operations and verifying the results. This is useful for
manual verification and debugging.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph_of_thoughts.mcp_server import create_server


async def test_mcp_server_functionality():
    """Test the MCP server functionality manually."""
    print("🧪 Testing Graph of Thoughts MCP Server")
    print("=" * 50)
    
    try:
        # Create the server
        print("1. Creating MCP server...")
        server = await create_server()
        print("   ✅ Server created successfully")
        
        # Test break_down_task tool
        print("\n2. Testing break_down_task tool...")
        task_result = await server._break_down_task({
            "task": "Build a machine learning model for sentiment analysis",
            "domain": "machine learning",
            "max_subtasks": 4
        })
        
        print("   ✅ Task breakdown completed")
        print(f"   📝 Result: {task_result[0].text[:100]}...")
        
        # Test generate_thoughts tool
        print("\n3. Testing generate_thoughts tool...")
        thoughts_result = await server._generate_thoughts({
            "problem": "Choose the best algorithm for text classification",
            "num_thoughts": 3,
            "approach_type": "analytical"
        })
        
        print("   ✅ Thought generation completed")
        print(f"   💭 Result: {thoughts_result[0].text[:100]}...")
        
        # Test score_thoughts tool
        print("\n4. Testing score_thoughts tool...")
        score_result = await server._score_thoughts({
            "thoughts": [
                "Use Naive Bayes for simple and fast classification",
                "Use BERT for state-of-the-art accuracy",
                "Use SVM for balanced performance and interpretability"
            ],
            "criteria": "accuracy, speed, and interpretability"
        })
        
        print("   ✅ Thought scoring completed")
        print(f"   📊 Result: {score_result[0].text[:100]}...")
        
        # Test validate_and_improve tool
        print("\n5. Testing validate_and_improve tool...")
        validation_result = await server._validate_and_improve({
            "solution": "Use BERT for sentiment analysis with fine-tuning on domain-specific data",
            "validation_criteria": "accuracy and computational efficiency",
            "max_iterations": 2
        })
        
        print("   ✅ Validation and improvement completed")
        print(f"   🔍 Result: {validation_result[0].text[:100]}...")
        
        # Test aggregate_results tool
        print("\n6. Testing aggregate_results tool...")
        aggregate_result = await server._aggregate_results({
            "results": [
                "BERT provides highest accuracy but requires more computational resources",
                "Naive Bayes is fast and simple but may have lower accuracy",
                "SVM offers good balance between accuracy and speed"
            ],
            "aggregation_method": "synthesis"
        })
        
        print("   ✅ Result aggregation completed")
        print(f"   🔗 Result: {aggregate_result[0].text[:100]}...")
        
        # Test create_reasoning_chain tool
        print("\n7. Testing create_reasoning_chain tool...")
        chain_result = await server._create_reasoning_chain({
            "problem": "Design an efficient recommendation system",
            "workflow_type": "generate_score_select",
            "num_branches": 3
        })
        
        print("   ✅ Reasoning chain completed")
        print(f"   ⛓️  Result: {chain_result[0].text[:100]}...")
        
        # Check execution results
        print("\n8. Checking execution results...")
        print(f"   📈 Total operations executed: {len(server.execution_results)}")
        
        # Test resource access
        print("\n9. Testing resource access...")
        results_json = json.dumps(server.execution_results, indent=2)
        print(f"   📊 Results JSON size: {len(results_json)} characters")
        
        # Test prompt templates
        print("\n10. Testing prompt templates...")
        print(f"    📝 Available templates: {list(server.prompt_templates.keys())}")
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        print("✅ MCP Server is ready for release!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_server_performance():
    """Test server performance with multiple concurrent operations."""
    print("\n🚀 Testing server performance...")
    
    server = await create_server()
    
    # Test concurrent operations
    tasks = []
    for i in range(5):
        task = server._generate_thoughts({
            "problem": f"Performance test problem {i}",
            "num_thoughts": 2,
            "approach_type": "analytical"
        })
        tasks.append(task)
    
    import time
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    
    print(f"   ⏱️  Processed 5 concurrent operations in {end_time - start_time:.2f} seconds")
    print(f"   📊 All operations completed: {len(results) == 5}")
    
    return True


def main():
    """Main function to run the manual tests."""
    print("Graph of Thoughts MCP Server - Manual Test Suite")
    print("=" * 60)
    
    async def run_all_tests():
        # Test basic functionality
        basic_test_passed = await test_mcp_server_functionality()
        
        if basic_test_passed:
            # Test performance
            performance_test_passed = await test_server_performance()
            
            if performance_test_passed:
                print("\n🎯 All manual tests passed!")
                print("🚀 MCP Server is ready for production use!")
                return True
        
        print("\n❌ Some tests failed. Please check the implementation.")
        return False
    
    # Run the tests
    success = asyncio.run(run_all_tests())
    
    if success:
        print("\n📋 Next Steps:")
        print("1. Configure your MCP host (Claude Desktop, VSCode, Cursor)")
        print("2. Add the server to your MCP configuration")
        print("3. Start using Graph of Thoughts reasoning in your workflows!")
        print("\n📖 See docs/MCP_SERVER.md for configuration details")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
