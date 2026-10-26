"""Tests for the MCP server tools, run in-memory against a small fixture dataset."""
import asyncio
import json
import os
import sys
from pathlib import Path

os.environ["RESTAURANTS_PATH"] = str(Path(__file__).parent / "fixtures" / "restaurants.json")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "concierge"))

from fastmcp import Client  # noqa: E402

from server import mcp  # noqa: E402


def call(tool: str, **args) -> dict:
    async def run():
        async with Client(mcp) as client:
            return (await client.call_tool(tool, args)).data

    return asyncio.run(run())


def test_lists_all_tools():
    async def run():
        async with Client(mcp) as client:
            return {tool.name for tool in await client.list_tools()}

    assert asyncio.run(run()) == {"search_restaurants", "get_restaurant", "get_reviews"}


def test_search_by_vibe_matches_tags_and_description():
    result = call("search_restaurants", vibe="moody")
    assert [r["name"] for r in result["results"]] == ["Ember & Ash", "Velvet Hour"]


def test_search_combines_filters_and_sorts_by_rating():
    in_la = call("search_restaurants", city="los angeles")
    assert [r["name"] for r in in_la["results"]] == ["Ember & Ash", "Velvet Hour"]

    affordable_in_la = call("search_restaurants", city="los angeles", max_price=3)
    assert [r["name"] for r in affordable_in_la["results"]] == ["Velvet Hour"]


def test_search_with_no_matches_suggests_cities():
    result = call("search_restaurants", cuisine="ethiopian")
    assert result["count"] == 0
    assert result["available_cities"] == ["Los Angeles", "San Diego"]


def test_get_restaurant_handles_partial_and_misspelled_names():
    assert call("get_restaurant", name="ember")["restaurant"]["name"] == "Ember & Ash"
    assert call("get_restaurant", name="Quiet Pyne")["restaurant"]["name"] == "Quiet Pine"
    assert "reviews" not in call("get_restaurant", name="Quiet Pine")["restaurant"]


def test_get_restaurant_not_found():
    assert call("get_restaurant", name="Nonexistent Diner")["found"] is False


def test_get_reviews_averages_ratings():
    result = call("get_reviews", name="Velvet Hour")
    assert result["average_review_rating"] == 4.0
    assert len(result["reviews"]) == 2


def test_overview_resource():
    async def run():
        async with Client(mcp) as client:
            return await client.read_resource("restaurants://overview")

    overview = json.loads(asyncio.run(run())[0].text)
    assert overview["total_restaurants"] == 3
    assert overview["cities"] == {"Los Angeles": 2, "San Diego": 1}
