"""
Weather Intelligence Agent
Uses Claude with MCP tools to provide intelligent weather-based
recommendations — not just weather data.
"""

import os
import json
import anthropic
from dotenv import load_dotenv
from mcp_server import MCP_TOOLS, execute_tool

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)
MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are an intelligent weather advisor that helps people 
make smart decisions based on real-time weather data.

You have access to weather tools that give you current conditions and forecasts
for any location worldwide. Always use these tools to get real data before
making recommendations.

Your role is NOT to just report weather. Your role is to:
- Understand what the person is actually trying to do
- Get the relevant weather data using your tools
- Reason about whether conditions support their goal
- Give a clear, specific recommendation with rationale
- Suggest alternatives if conditions aren't ideal

Examples of good responses:
- "Today is great for your 5K — 62°F with low humidity and a light breeze. 
   Go between 7-9am before it warms up."
- "Chicago Monday will be fine for meetings but pack an umbrella. Miami 
   Wednesday looks perfect — 78°F and sunny."
- "Spain next month varies by region. Madrid will be hot (95°F+), coastal 
   Barcelona much more comfortable at 80°F. Which cities are you visiting?"

Always be specific, practical, and actionable. Think like a smart friend 
who happens to have real-time weather data, not like a weather app."""


async def process_query(user_message: str) -> dict:
    """
    Process a user query using Claude with MCP weather tools.
    Returns the final response and tool calls made.
    """
    messages = [{"role": "user", "content": user_message}]
    tool_calls_made = []
    
    # Agentic loop — Claude calls tools until it has enough data
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=MCP_TOOLS,
            messages=messages
        )
        
        # If Claude wants to use a tool
        if response.stop_reason == "tool_use":
            tool_results = []
            
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    
                    print(f"  → Calling tool: {tool_name}({tool_input})")
                    
                    # Execute the tool via MCP server
                    result = await execute_tool(tool_name, tool_input)
                    
                    tool_calls_made.append({
                        "tool": tool_name,
                        "input": tool_input,
                        "result_preview": result[:200] + "..." if len(result) > 200 else result
                    })
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            # Add Claude's response and tool results to conversation
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            
        # Claude has finished reasoning and has a final answer
        elif response.stop_reason == "end_turn":
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text
            
            return {
                "response": final_text,
                "tool_calls": tool_calls_made,
                "tools_used": len(tool_calls_made)
            }
        
        else:
            break
    
    return {
        "response": "Unable to process request.",
        "tool_calls": [],
        "tools_used": 0
    }