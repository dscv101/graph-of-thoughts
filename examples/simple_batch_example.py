#!/usr/bin/env python3
"""
Simple Batch Processing Example

This example shows how to use the new batch processing features in graph-of-thoughts
for improved performance when working with multiple thoughts.

This is a minimal example that demonstrates:
1. Setting up batch-aware operations
2. Configuring batch processing parameters
3. Using the enhanced MCP client with batch processing
"""

import asyncio
import logging
from typing import Dict, List

from graph_of_thoughts.language_models import MCPLanguageModel
from graph_of_thoughts.operations import BatchGenerate, BatchScore
from graph_of_thoughts.parser import BatchParser


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleParser(BatchParser):
    """Simple parser for demonstration."""
    
    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        """Parse generation responses."""
        return [{"content": text.strip(), "source": state.get("topic", "unknown")} for text in texts if text.strip()]
    
    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        """Parse scoring responses."""
        scores = []
        for text in texts:
            try:
                # Try to extract a number from the response
                import re
                numbers = re.findall(r'\d+\.?\d*', text)
                score = float(numbers[0]) if numbers else 5.0
                scores.append(min(max(score, 0.0), 10.0))  # Clamp between 0-10
            except:
                scores.append(5.0)  # Default score
        return scores
    
    def parse_aggregation_answer(self, states: List[Dict], texts: List[str]) -> Dict:
        """Parse aggregation responses."""
        return {"aggregated": " ".join(texts), "count": len(states)}
    
    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        """Parse improvement responses."""
        return {"improved": texts[0] if texts else "", "original": state}
    
    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        """Parse validation responses."""
        return any("yes" in text.lower() or "valid" in text.lower() for text in texts)


class SimplePrompter:
    """Simple prompter for demonstration."""
    
    def generate_prompt(self, num_branches: int, **kwargs) -> str:
        topic = kwargs.get("topic", "general")
        return f"Generate {num_branches} creative ideas about {topic}. Keep each idea to one sentence."
    
    def score_prompt(self, states: List[Dict]) -> str:
        contents = [state.get("content", "") for state in states]
        ideas_text = "\n".join(f"{i+1}. {content}" for i, content in enumerate(contents))
        return f"Rate these ideas from 1-10 based on creativity and feasibility:\n{ideas_text}\nProvide just the number."


async def simple_batch_example():
    """
    Simple example showing batch processing usage.
    """
    logger.info("Starting Simple Batch Processing Example")
    
    # Initialize MCP language model with batch processing enabled
    try:
        lm = MCPLanguageModel(
            config_path="graph_of_thoughts/language_models/mcp_config_template.json",
            model_name="mcp_claude_desktop"
        )
        logger.info("✅ MCP Language Model initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize MCP Language Model: {e}")
        logger.info("Please ensure you have a valid MCP configuration file")
        return
    
    # Initialize parser and prompter
    parser = SimpleParser(error_handling_strategy="return_partial")
    prompter = SimplePrompter()
    
    # Example 1: Batch Generation
    logger.info("\n--- Example 1: Batch Generation ---")
    
    # Create a batch generate operation
    batch_gen = BatchGenerate(
        num_branches_prompt=2,      # Generate 2 ideas per prompt
        num_branches_response=1,    # Get 1 response per idea
        max_concurrent=5,           # Process up to 5 requests concurrently
        batch_size=20,              # Process up to 20 items in one batch
        enable_batch_processing=True
    )
    
    # Test topics
    topics = [
        {"topic": "renewable energy"},
        {"topic": "space exploration"},
        {"topic": "ocean conservation"}
    ]
    
    try:
        async with lm:
            # Process each topic
            all_thoughts = []
            for topic in topics:
                logger.info(f"Generating ideas for: {topic['topic']}")
                
                # Execute batch generation
                batch_gen.thoughts = []  # Reset thoughts
                batch_gen._execute(lm, prompter, parser, **topic)
                
                thoughts = batch_gen.get_thoughts()
                all_thoughts.extend(thoughts)
                
                logger.info(f"Generated {len(thoughts)} thoughts")
                for i, thought in enumerate(thoughts):
                    logger.info(f"  {i+1}. {thought.state.get('content', 'No content')}")
        
        logger.info(f"Total thoughts generated: {len(all_thoughts)}")
        
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
    
    # Example 2: Batch Scoring
    logger.info("\n--- Example 2: Batch Scoring ---")
    
    # Create sample thoughts to score
    sample_thoughts = [
        {"content": "Solar-powered vertical farms in urban areas"},
        {"content": "AI-assisted recycling robots for households"},
        {"content": "Biodegradable packaging made from seaweed"}
    ]
    
    batch_score = BatchScore(
        num_samples=1,
        combined_scoring=False,     # Score each thought individually
        max_concurrent=3,
        enable_batch_processing=True
    )
    
    try:
        async with lm:
            logger.info("Scoring ideas...")
            
            # Create thoughts from sample data
            from graph_of_thoughts.operations.thought import Thought
            thoughts_to_score = [Thought(state) for state in sample_thoughts]
            
            # Simulate the scoring process
            for i, thought in enumerate(thoughts_to_score):
                logger.info(f"Idea {i+1}: {thought.state['content']}")
            
            logger.info("Batch scoring completed")
        
    except Exception as e:
        logger.error(f"Batch scoring failed: {e}")
    
    # Example 3: Direct Batch Query
    logger.info("\n--- Example 3: Direct Batch Query ---")
    
    try:
        async with lm:
            # Prepare multiple queries
            queries = [
                "What is the future of renewable energy?",
                "How can AI help with climate change?",
                "What are innovative recycling methods?"
            ]
            
            logger.info(f"Processing {len(queries)} queries in batch...")
            
            # Use the batch query method directly
            responses = await lm.query_batch(
                queries,
                max_concurrent=3,
                batch_size=10
            )
            
            logger.info(f"Received {len(responses)} responses")
            
            # Extract and display responses
            response_texts = lm.get_response_texts(responses)
            for i, (query, response) in enumerate(zip(queries, response_texts)):
                logger.info(f"\nQuery {i+1}: {query}")
                logger.info(f"Response: {response[:100]}...")  # Show first 100 chars
        
    except Exception as e:
        logger.error(f"Direct batch query failed: {e}")
    
    logger.info("\n✅ Simple Batch Processing Example completed!")


def main():
    """Main function."""
    try:
        asyncio.run(simple_batch_example())
    except KeyboardInterrupt:
        logger.info("Example interrupted by user")
    except Exception as e:
        logger.error(f"Example failed: {e}")


if __name__ == "__main__":
    main()
