"""
MCP Weather Intelligence Server
Author: Dujuan Brown | Iron Gate Solutions US LLC
GitHub: github.com/JDPPW/db-ai-portfolio

Custom MCP server that exposes real-time weather tools to Claude.
The agent uses these tools during reasoning to make intelligent
decisions — not just return raw weather data.

This demonstrates Model Context Protocol (MCP) — the standard
for connecting AI agents to external tools and data sources.

Target roles: WorldVia, GreenLight, Conduet, Kyndryl, Trilagen
IBM RAG & Agentic AI Certificate — Course 9 (MCP)
"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
WEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

# ── Weather tool functions exposed via MCP ────────────────────────────────────

async def get_current_weather(location: str) -> dict:
    """
    Get current weather conditions for any location.
    Returns temperature, humidity, wind, conditions.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{WEATHER_BASE_URL}/weather",
            params={
                "q": location,
                "appid": WEATHER_API_KEY,
                "units": "imperial"
            }
        )

        if response.status_code == 404:
            return {"error": f"Location '{location}' not found. Try a more specific city name."}

        if response.status_code != 200:
            return {"error": f"Weather API error: {response.status_code}"}

        data = response.json()

        return {
            "location": f"{data['name']}, {data['sys']['country']}",
            "temperature_f": round(data["main"]["temp"]),
            "feels_like_f": round(data["main"]["feels_like"]),
            "humidity_percent": data["main"]["humidity"],
            "wind_mph": round(data["wind"]["speed"]),
            "conditions": data["weather"][0]["description"].title(),
            "visibility_miles": round(data.get("visibility", 0) / 1609),
            "pressure_hpa": data["main"]["pressure"]
        }


async def get_forecast(location: str, days: int = 5) -> dict:
    """
    Get 5-day weather forecast for any location.
    Returns daily summaries with temperature ranges and conditions.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{WEATHER_BASE_URL}/forecast",
            params={
                "q": location,
                "appid": WEATHER_API_KEY,
                "units": "imperial",
                "cnt": days * 8  # API returns data in 3-hour intervals
            }
        )

        if response.status_code == 404:
            return {"error": f"Location '{location}' not found."}

        if response.status_code != 200:
            return {"error": f"Weather API error: {response.status_code}"}

        data = response.json()

        # Group forecasts by day
        daily = {}
        for item in data["list"]:
            date = item["dt_txt"].split(" ")[0]
            if date not in daily:
                daily[date] = {
                    "temps": [],
                    "conditions": [],
                    "humidity": [],
                    "wind": [],
                    "rain": False
                }
            daily[date]["temps"].append(item["main"]["temp"])
            daily[date]["conditions"].append(item["weather"][0]["description"])
            daily[date]["humidity"].append(item["main"]["humidity"])
            daily[date]["wind"].append(item["wind"]["speed"])
            if "rain" in item:
                daily[date]["rain"] = True

        # Summarize each day
        forecast_days = []
        for date, d in list(daily.items())[:days]:
            forecast_days.append({
                "date": date,
                "high_f": round(max(d["temps"])),
                "low_f": round(min(d["temps"])),
                "avg_humidity": round(sum(d["humidity"]) / len(d["humidity"])),
                "avg_wind_mph": round(sum(d["wind"]) / len(d["wind"])),
                "conditions": max(set(d["conditions"]), key=d["conditions"].count).title(),
                "rain_expected": d["rain"]
            })

        return {
            "location": f"{data['city']['name']}, {data['city']['country']}",
            "forecast": forecast_days
        }


async def compare_locations(locations: list) -> dict:
    """
    Compare weather across multiple locations simultaneously.
    Used for multi-city trip planning.
    """
    results = {}
    for location in locations:
        current = await get_current_weather(location)
        forecast = await get_forecast(location, days=3)
        results[location] = {
            "current": current,
            "forecast": forecast.get("forecast", [])
        }
    return results


# ── MCP tool definitions ──────────────────────────────────────────────────────
# These are the tool schemas Claude uses to understand what tools are available
# and how to call them

MCP_TOOLS = [
    {
        "name": "get_current_weather",
        "description": "Get current weather conditions for any city or location worldwide. Returns temperature, humidity, wind speed, and conditions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, optionally with country code. Examples: 'London', 'Paris, FR', 'New York, US', 'Tokyo, JP'"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "get_forecast",
        "description": "Get 5-day weather forecast for any location. Returns daily high/low temperatures, conditions, humidity, wind, and rain probability. Use this to recommend the best days for activities or travel.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, optionally with country code."
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days to forecast (1-5). Default is 5.",
                    "default": 5
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "compare_locations",
        "description": "Compare weather across multiple locations simultaneously. Use this when someone is planning a multi-city trip or needs to compare weather in different places.",
        "input_schema": {
            "type": "object",
            "properties": {
                "locations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of city names to compare. Example: ['New York', 'Los Angeles', 'Chicago']"
                }
            },
            "required": ["locations"]
        }
    }
]


# ── Tool executor ─────────────────────────────────────────────────────────────

async def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool call and return result as string for Claude."""
    import json

    if tool_name == "get_current_weather":
        result = await get_current_weather(tool_input["location"])
    elif tool_name == "get_forecast":
        result = await get_forecast(
            tool_input["location"],
            tool_input.get("days", 5)
        )
    elif tool_name == "compare_locations":
        result = await compare_locations(tool_input["locations"])
    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    return json.dumps(result, indent=2)