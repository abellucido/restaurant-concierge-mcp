"""MCP server exposing a California restaurant dataset as tools and resources.

Run it directly (stdio transport) or let the chatbot start it:
    python src/concierge/server.py
"""
import difflib
import json
import os
from collections import Counter
from pathlib import Path

from fastmcp import FastMCP

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "restaurants.json"

mcp = FastMCP("restaurant-concierge")


def load_restaurants() -> list[dict]:
    """Load the dataset (path can be overridden with RESTAURANTS_PATH)."""
    path = Path(os.environ.get("RESTAURANTS_PATH", DEFAULT_DATA_PATH))
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(restaurant: dict) -> dict:
    """Compact view used in search results, so the LLM context stays small."""
    return {
        "name": restaurant["name"],
        "cuisine": restaurant["cuisine"],
        "location": f"{restaurant['neighborhood']}, {restaurant['city']}",
        "price_range": restaurant["price_range"],
        "rating": restaurant["rating"],
        "vibes": restaurant["vibes"],
    }


def find_by_name(name: str, restaurants: list[dict]) -> dict | None:
    """Match a restaurant by exact, partial, then fuzzy name."""
    query = name.lower().strip()
    by_name = {r["name"].lower(): r for r in restaurants}

    if query in by_name:
        return by_name[query]
    partial = [r for key, r in by_name.items() if query in key or key in query]
    if partial:
        return partial[0]
    close = difflib.get_close_matches(query, by_name, n=1, cutoff=0.6)
    return by_name[close[0]] if close else None


@mcp.tool()
def search_restaurants(
    cuisine: str | None = None,
    city: str | None = None,
    vibe: str | None = None,
    max_price: int | None = None,
    min_rating: float | None = None,
    limit: int = 5,
) -> dict:
    """Search restaurants by any combination of filters.

    Results are sorted by rating (best first), so to find the best-rated
    restaurant just search without min_rating and take the first result.
    Only set filters the user actually asked for.

    Args:
        cuisine: Cuisine keyword, e.g. "italian", "sushi", "vegan".
        city: City or neighborhood, e.g. "San Francisco", "Little Tokyo".
        vibe: Atmosphere keyword, e.g. "moody", "romantic", "zen", "family-friendly".
        max_price: Highest price level from 1 ($) to 4 ($$$$).
        min_rating: Lowest acceptable rating, from 1 to 5.
        limit: Maximum number of results (default 5).
    """
    restaurants = load_restaurants()
    matches = []
    for r in restaurants:
        if cuisine and cuisine.lower() not in r["cuisine"].lower():
            continue
        if city and city.lower() not in f"{r['city']} {r['neighborhood']}".lower():
            continue
        if vibe:
            vibe_text = " ".join(r["vibes"] + [r["description"]]).lower()
            if vibe.lower() not in vibe_text:
                continue
        if max_price and len(r["price_range"]) > max_price:
            continue
        if min_rating and r["rating"] < min_rating:
            continue
        matches.append(r)

    if not matches:
        # Give the model what it needs to retry with a better query
        return {
            "count": 0,
            "results": [],
            "hint": "No matches. Retry with fewer filters, or use one of the available cities.",
            "available_cities": sorted({r["city"] for r in restaurants}),
        }

    matches.sort(key=lambda r: r["rating"], reverse=True)
    return {
        "count": len(matches),
        "results": [summarize(r) for r in matches[:limit]],
    }


@mcp.tool()
def get_restaurant(name: str) -> dict:
    """Get full details for one restaurant: description, signature dishes,
    price, rating and vibes. Partial or slightly misspelled names work."""
    restaurants = load_restaurants()
    restaurant = find_by_name(name, restaurants)
    if not restaurant:
        return {"found": False, "message": f"No restaurant matches '{name}'."}

    details = {key: value for key, value in restaurant.items() if key != "reviews"}
    return {"found": True, "restaurant": details}


@mcp.tool()
def get_reviews(name: str) -> dict:
    """Get customer reviews and the average review score for a restaurant."""
    restaurant = find_by_name(name, load_restaurants())
    if not restaurant:
        return {"found": False, "message": f"No restaurant matches '{name}'."}

    reviews = restaurant["reviews"]
    return {
        "found": True,
        "restaurant": restaurant["name"],
        "average_review_rating": round(sum(r["rating"] for r in reviews) / len(reviews), 1),
        "reviews": reviews,
    }


@mcp.resource("restaurants://overview")
def overview() -> str:
    """Which cities, cuisines and vibes the dataset covers."""
    restaurants = load_restaurants()
    return json.dumps(
        {
            "total_restaurants": len(restaurants),
            "cities": dict(Counter(r["city"] for r in restaurants).most_common()),
            "cuisines": sorted({r["cuisine"] for r in restaurants}),
            "vibes": dict(Counter(v.lower() for r in restaurants for v in r["vibes"]).most_common(15)),
        },
        indent=2,
    )


if __name__ == "__main__":
    mcp.run()
