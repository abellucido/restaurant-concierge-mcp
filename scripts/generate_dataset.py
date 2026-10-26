"""Generate a fictional California restaurant dataset with a local Ollama model.

Each restaurant is produced from a (city, neighborhood, cuisine, vibe, price) seed
so the dataset covers a spread of places and atmospheres. The model writes the
text, constrained to a JSON schema and validated with Pydantic. Numbers (price
and ratings) are assigned in code, because LLMs pick nearly identical values
every time.

Usage:
    python scripts/generate_dataset.py               # all seeds
    python scripts/generate_dataset.py --limit 5     # quick trial run
    python scripts/generate_dataset.py --renumber    # reassign numbers only
"""
import argparse
import json
import os
import random
from pathlib import Path

import ollama
from pydantic import BaseModel, Field, ValidationError

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "restaurants.json"
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
MAX_ATTEMPTS = 3

# (city, neighborhood, cuisine, vibe, price_range)
SEEDS = [
    ("Los Angeles", "Arts District", "Wood-fired steakhouse", "moody", "$$$$"),
    ("Los Angeles", "Little Tokyo", "Japanese omakase", "zen", "$$$$"),
    ("Los Angeles", "Koreatown", "Korean barbecue", "lively", "$$"),
    ("Los Angeles", "Silver Lake", "Californian farm-to-table", "bohemian", "$$$"),
    ("Los Angeles", "Highland Park", "Oaxacan", "colorful", "$$"),
    ("Los Angeles", "Venice", "Vegan bowls", "sun-drenched", "$"),
    ("Los Angeles", "Chinatown", "Cantonese dim sum", "bustling", "$$"),
    ("Los Angeles", "Downtown", "Northern Italian", "romantic", "$$$"),
    ("Santa Monica", "Main Street", "Baja seafood", "sun-drenched", "$"),
    ("Pasadena", "Old Town", "French bistro", "romantic", "$$$"),
    ("Long Beach", "Belmont Shore", "Cajun and Creole", "moody", "$$"),
    ("San Diego", "Little Italy", "Neapolitan pizza", "family-friendly", "$$"),
    ("San Diego", "North Park", "Craft brewery gastropub", "lively", "$$"),
    ("San Diego", "La Jolla", "Coastal seafood", "romantic", "$$$$"),
    ("San Francisco", "Mission District", "Mission-style taqueria", "casual", "$"),
    ("San Francisco", "North Beach", "Old-school Italian", "cozy", "$$"),
    ("San Francisco", "Richmond District", "Burmese", "cozy", "$$"),
    ("San Francisco", "SoMa", "Modern Filipino", "trendy", "$$$"),
    ("San Francisco", "Japantown", "Ramen", "casual", "$"),
    ("Oakland", "Temescal", "Ethiopian", "family-friendly", "$$"),
    ("Oakland", "Uptown", "Southern soul food", "lively", "$$"),
    ("Berkeley", "Gourmet Ghetto", "Vegetarian tasting menu", "zen", "$$$$"),
    ("San Jose", "Japantown", "Vietnamese pho", "casual", "$"),
    ("Sacramento", "Midtown", "Farm-to-fork New American", "rustic", "$$$"),
    ("Napa", "Downtown Napa", "Wine-country Californian", "romantic", "$$$$"),
    ("Healdsburg", "The Plaza", "Mediterranean small plates", "rustic", "$$$"),
    ("Carmel-by-the-Sea", "Ocean Avenue", "Japanese kaiseki", "zen", "$$$$"),
    ("Monterey", "Cannery Row", "Seafood chowder house", "casual", "$$"),
    ("Santa Barbara", "Funk Zone", "Spanish tapas", "trendy", "$$$"),
    ("Palm Springs", "Uptown Design District", "Mid-century cocktail lounge", "retro", "$$$"),
    ("Big Sur", "Highway 1", "Cliffside Californian", "rustic", "$$$"),
    ("Fresno", "Tower District", "Armenian", "family-friendly", "$$"),
    ("Irvine", "Diamond Jamboree", "Taiwanese", "bustling", "$"),
    ("Anaheim", "Little Arabia", "Lebanese", "family-friendly", "$"),
    ("Truckee", "Historic Downtown", "Mountain lodge grill", "cozy", "$$$"),
    ("Los Angeles", "West Hollywood", "Speakeasy small plates", "moody", "$$$"),
]


class Review(BaseModel):
    reviewer: str = Field(max_length=40, description="A made-up first name and last initial, e.g. 'Maya R.'")
    rating: float = Field(ge=1, le=5, description="Lower for more critical reviews")
    text: str = Field(max_length=500, description="Two or three sentences in a natural, personal voice")


class Restaurant(BaseModel):
    name: str = Field(max_length=60)
    cuisine: str = Field(max_length=60)
    city: str = Field(max_length=60)
    neighborhood: str = Field(max_length=60)
    vibes: list[str] = Field(min_length=2, max_length=5)
    description: str = Field(max_length=700, description="Three or four vivid sentences about the space, food and atmosphere")
    signature_dishes: list[str] = Field(min_length=2, max_length=4)
    reviews: list[Review] = Field(min_length=2, max_length=3)


PRICE_WORDS = {"$": "cheap", "$$": "moderately priced", "$$$": "upscale", "$$$$": "very expensive"}


def build_prompt(city, neighborhood, cuisine, vibe, price_range, taken_names):
    return f"""Invent a fictional restaurant for a dataset of California restaurants.

- Location: {neighborhood}, {city}, California
- Cuisine: {cuisine}
- Overall vibe: {vibe} (include "{vibe}" in the vibes list, plus 1-4 related words)
- Price level: {PRICE_WORDS[price_range]}

Give it an original, memorable name that is not a real restaurant and is not one of: {", ".join(taken_names) or "none yet"}.
Include 2-3 short customer reviews with varied opinions, including at least one mild criticism.
Respond with JSON only."""


def assign_numbers(record: dict, seed_index: int) -> dict:
    """Set price from the seed and spread ratings deterministically.

    Review ratings vary around the restaurant's rating, keeping the model's
    ordering so the most critical review still gets the lowest score.
    """
    rng = random.Random(seed_index)
    rating = round(rng.uniform(3.6, 4.9), 1)

    reviews = record["reviews"]
    offsets = sorted(rng.uniform(-0.9, 0.5) for _ in reviews)
    by_model_score = sorted(range(len(reviews)), key=lambda i: reviews[i]["rating"])
    for offset, i in zip(offsets, by_model_score):
        reviews[i]["rating"] = min(5.0, max(1.0, round((rating + offset) * 2) / 2))

    record["price_range"] = SEEDS[seed_index][4]
    record["rating"] = rating
    return record


def generate_restaurant(seed, taken_names):
    city, neighborhood = seed[0], seed[1]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": build_prompt(*seed, taken_names)}],
            format=Restaurant.model_json_schema(),
            # num_predict caps runaway generations (a full record is ~400 tokens)
            options={"temperature": 0.8, "num_predict": 900},
        )
        try:
            restaurant = Restaurant.model_validate_json(response.message.content)
        except ValidationError as error:
            print(f"  attempt {attempt}: invalid output ({error.error_count()} errors), retrying", flush=True)
            continue

        if restaurant.name.lower() in (n.lower() for n in taken_names):
            print(f"  attempt {attempt}: duplicate name '{restaurant.name}', retrying", flush=True)
            continue

        # Keep the seed's location authoritative in case the model drifted
        restaurant.city, restaurant.neighborhood = city, neighborhood
        return restaurant

    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=len(SEEDS), help="number of seeds to use")
    parser.add_argument("--renumber", action="store_true",
                        help="reassign price and ratings in the existing file without calling the model")
    args = parser.parse_args()

    if args.renumber:
        restaurants = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        for record in restaurants:
            seed_index = next(i for i, s in enumerate(SEEDS) if (s[0], s[1], s[2]) ==
                              (record["city"], record["neighborhood"], record["cuisine"]))
            assign_numbers(record, seed_index)
        save(restaurants)
        print(f"Renumbered {len(restaurants)} restaurants in {OUTPUT_PATH}")
        return

    restaurants = []
    for i, seed in enumerate(SEEDS[: args.limit]):
        print(f"[{i + 1}/{args.limit}] {seed[2]} in {seed[1]}, {seed[0]}", flush=True)
        restaurant = generate_restaurant(seed, [r["name"] for r in restaurants])
        if restaurant is None:
            print("  skipped after repeated failures", flush=True)
            continue
        record = assign_numbers({"id": len(restaurants) + 1, **restaurant.model_dump()}, i)
        restaurants.append(record)
        print(f"  -> {restaurant.name}", flush=True)

        # Save after every restaurant so an interrupted run keeps its progress
        save(restaurants)

    print(f"Saved {len(restaurants)} restaurants to {OUTPUT_PATH}")


def save(restaurants: list[dict]):
    """Write the dataset with a consistent key order."""
    order = ["id", "name", "cuisine", "city", "neighborhood", "price_range", "rating",
             "vibes", "description", "signature_dishes", "reviews"]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ordered = [{key: r[key] for key in order} for r in restaurants]
    OUTPUT_PATH.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
