---

title: "Tool Decorators for AI Agents: Keeping Code and Metadata in Sync"
date: "2026-03-09 10:00:00 +0530"
categories: [AgenticAI, Decorators]
tags: [AgenticAI, Decorators]
author: Divyesh Rajpura

---


# Tool Decorators for AI Agents: Keeping Code and Metadata in Sync

> *How Python decorators eliminate the most common source of agent bugs — and make your tools self-documenting*

---

In the [previous post](https://divyeshrajpura4114.github.io/posts/agentic-ai-game-components/), we built a Hotel Finder Agent using the GAME framework. Every action — `geocode_city`, `search_hotels`, `terminate` — was manually registered with its name, description, and parameter schema written out by hand.

That works. Until you change something.

Add a new parameter to `search_hotels`? You need to update the schema. Improve the description? You need to find it in the registration code, separate from the function. Over time, these two things — the function and its metadata — drift apart. The agent starts getting the wrong instructions about its own tools. Bugs follow.

This post introduces **tool decorators** — a pattern that makes your functions the single source of truth for everything the agent needs to know about them.

---

## The Problem: Two Places to Maintain the Same Thing

Here is how our hotel finder registered `search_hotels` in the previous approach:

```python
def search_hotels(lat: float, lon: float, radius_km: float = 5) -> List[Dict]:
    # ... implementation ...

# Separately, in registration code:
action_registry.register(Action(
    name="search_hotels",
    function=search_hotels,
    description="Searches for hotels near a lat/lon coordinate using the Overpass API.",
    parameters={
        "type": "object",
        "properties": {
            "lat":       {"type": "number"},
            "lon":       {"type": "number"},
            "radius_km": {"type": "number"},
        },
        "required": ["lat", "lon"]
    },
    terminal=False
))
```

This pattern has three failure modes:

**Schema drift** — You add `min_stars: int = None` to the function but forget to add it to `parameters`. The agent never knows it can filter by stars.

**Description rot** — You change what the function does but the description still says the old thing. The agent makes decisions based on stale documentation.

**Registration noise** — Every new tool requires a block of boilerplate. The more tools you have, the harder it is to see what's actually different between them.

The fix is to make the function itself the authoritative source for all of this — name, description, parameters, and registration. That is exactly what a decorator does.

---

## The Solution: `@register_tool`

With decorators, the hotel finder tools look like this:

```python
@register_tool(tags=["geocode"])
def geocode_city(city: str) -> Dict:
    """Resolve a city name to lat/lon via Nominatim (free, no API key).

    Always call this before search_hotels. Returns the full display name
    of the city along with its latitude and longitude coordinates.

    Args:
        city: City name to geocode, e.g. 'Amsterdam' or 'Tokyo, Japan'

    Returns:
        A dict with keys: city (display name), lat (float), lon (float)
    """
    resp = requests.get(
        NOMINATIM_URL,
        params={"q": city, "format": "json", "limit": 1},
        headers=HEADERS,
        timeout=10,
    )
    results = resp.json()
    if not results:
        return {"error": f"City '{city}' not found"}
    r = results[0]
    return {"city": r["display_name"], "lat": float(r["lat"]), "lon": float(r["lon"])}


@register_tool(tags=["search", "hotels"])
def search_hotels(lat: float, lon: float, radius_km: float = 5) -> List[Dict]:
    """Fetch hotels from OpenStreetMap via Overpass API (free, no API key).

    Searches within a circular area around the given coordinates. Returns
    hotels with their name, type, star rating, pet policy, wifi, and address.
    Use radius_km=3 for dense city centres, up to 10 for rural areas.

    Args:
        lat: Latitude from geocode_city
        lon: Longitude from geocode_city
        radius_km: Search radius in kilometres (default 5)

    Returns:
        List of hotel dicts with keys: name, type, stars, pets, wifi, address
    """
    radius_m = int(radius_km * 1000)
    query = f"""
[out:json][timeout:30];
(
  node["tourism"="hotel"](around:{radius_m},{lat},{lon});
  way["tourism"="hotel"](around:{radius_m},{lat},{lon});
  node["tourism"="motel"](around:{radius_m},{lat},{lon});
  node["tourism"="guest_house"](around:{radius_m},{lat},{lon});
  node["tourism"="hostel"](around:{radius_m},{lat},{lon});
);
out body center 60;
""".strip()
    resp = requests.post(OVERPASS_URL, data=query, timeout=30)
    hotels = []
    for el in resp.json().get("elements", []):
        tags = el.get("tags", {})
        if not tags.get("name"):
            continue
        hotels.append({
            "name":    tags.get("name"),
            "type":    tags.get("tourism", "hotel"),
            "stars":   tags.get("stars") or tags.get("stars:official"),
            "pets":    tags.get("pets") or tags.get("dog") or tags.get("pets_allowed"),
            "wifi":    tags.get("internet_access"),
            "address": " ".join(filter(None, [
                           tags.get("addr:housenumber"),
                           tags.get("addr:street"),
                           tags.get("addr:city"),
                       ])),
        })
    return hotels


@register_tool(tags=["system"], terminal=True)
def terminate(message: str) -> str:
    """Deliver the final hotel recommendations to the user.

    Call this once you have ranked the hotels and selected the top 5.
    The message should include a numbered list with a brief explanation
    for each hotel covering how it matches the user's criteria.

    Args:
        message: The final ranked hotel recommendations
    """
    return message
```

Now there is one place to look, one place to update, and nothing can fall out of sync.

---

## How It Works: Under the Hood

The `@register_tool` decorator is built from two functions working together.

### `get_tool_metadata` — The Inspector

This helper does the actual introspection. Given any Python function, it extracts everything needed to describe the tool to an LLM:

```python
import inspect
from typing import get_type_hints

def get_tool_metadata(func,
                      tool_name=None,
                      description=None,
                      parameters_override=None,
                      terminal=False,
                      tags=None):
    # 1. Name: use the provided name, or fall back to the function name
    tool_name = tool_name or func.__name__

    # 2. Description: use the provided description, or extract the docstring
    description = description or (func.__doc__.strip() if func.__doc__ else "No description provided.")

    # 3. Parameters: introspect the function signature and type hints
    if parameters_override is None:
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)

        args_schema = {"type": "object", "properties": {}, "required": []}

        for param_name, param in signature.parameters.items():
            if param_name in ["action_context", "action_agent"]:
                continue  # Skip internal framework parameters

            param_type = type_hints.get(param_name, str)
            args_schema["properties"][param_name] = {"type": get_json_type(param_type)}

            # A parameter without a default value is required
            if param.default == inspect.Parameter.empty:
                args_schema["required"].append(param_name)
    else:
        args_schema = parameters_override

    return {
        "tool_name":   tool_name,
        "description": description,
        "parameters":  args_schema,
        "function":    func,
        "terminal":    terminal,
        "tags":        tags or [],
    }
```

For `search_hotels`, this produces:

```python
{
    "tool_name": "search_hotels",
    "description": "Fetch hotels from OpenStreetMap via Overpass API...",
    "parameters": {
        "type": "object",
        "properties": {
            "lat":       {"type": "number"},
            "lon":       {"type": "number"},
            "radius_km": {"type": "number"},
        },
        "required": ["lat", "lon"]   # radius_km has a default, so it's optional
    },
    "terminal": False,
    "tags": ["search", "hotels"]
}
```

Notice that `lat` and `lon` are automatically marked as required because they have no default values. `radius_km` is optional because it defaults to `5`. The decorator figured this out from the function signature — you wrote nothing extra.

### `register_tool` — The Decorator Factory

This is the outer layer that wraps `get_tool_metadata` and registers the result in two global dictionaries:

```python
tools = {}           # All tools, indexed by name
tools_by_tag = {}    # Tool names grouped by tag

def register_tool(tool_name=None, description=None,
                  parameters_override=None, terminal=False, tags=None):
    def decorator(func):
        metadata = get_tool_metadata(
            func=func,
            tool_name=tool_name,
            description=description,
            parameters_override=parameters_override,
            terminal=terminal,
            tags=tags,
        )

        # Register in the main tools dictionary
        tools[metadata["tool_name"]] = {
            "description": metadata["description"],
            "parameters":  metadata["parameters"],
            "function":    metadata["function"],
            "terminal":    metadata["terminal"],
            "tags":        metadata["tags"],
        }

        # Register in the tag index for easy lookup
        for tag in metadata["tags"]:
            if tag not in tools_by_tag:
                tools_by_tag[tag] = []
            tools_by_tag[tag].append(metadata["tool_name"])

        return func  # Return the original function unchanged
    return decorator
```

The last line — `return func` — is important. The decorator registers the function as a tool but doesn't change the function itself. You can still call `geocode_city("Amsterdam")` directly in tests or scripts, exactly as before.

---

## Organizing Tools with Tags

Tags turn the flat `tools` dictionary into a structured library. The `tools_by_tag` index lets you look up tools by what they do rather than what they're called.

For the hotel finder, our tags look like this after registration:

```python
tools_by_tag = {
    "geocode":  ["geocode_city"],
    "search":   ["search_hotels"],
    "hotels":   ["search_hotels"],
    "system":   ["terminate"],
}
```

Think of tags like labels in a drawer. A tool can have multiple tags because it might belong to multiple categories. `search_hotels` is tagged both `"search"` (what kind of operation) and `"hotels"` (what domain).

### `PythonActionRegistry` — Tag-Aware Loading

The `PythonActionRegistry` extends the base `ActionRegistry` to load tools directly from the `tools` dictionary, filtered by tags:

```python
class PythonActionRegistry(ActionRegistry):
    def __init__(self, tags: List[str] = None, tool_names: List[str] = None):
        super().__init__()
        self.terminate_tool = None

        for tool_name, tool_desc in tools.items():
            # Always track the terminate tool separately
            if tool_name == "terminate":
                self.terminate_tool = tool_desc

            # Skip if not in the requested tool_names list
            if tool_names and tool_name not in tool_names:
                continue

            # Skip if none of the tool's tags match the requested tags
            tool_tags = tool_desc.get("tags", [])
            if tags and not any(tag in tool_tags for tag in tags):
                continue

            self.register(Action(
                name=tool_name,
                function=tool_desc["function"],
                description=tool_desc["description"],
                parameters=tool_desc.get("parameters", {}),
                terminal=tool_desc.get("terminal", False),
            ))

    def register_terminate_tool(self):
        """Explicitly add the terminate tool if not already included by tags."""
        if self.terminate_tool:
            self.register(Action(
                name="terminate",
                function=self.terminate_tool["function"],
                description=self.terminate_tool["description"],
                parameters=self.terminate_tool.get("parameters", {}),
                terminal=True,
            ))
        else:
            raise Exception("Terminate tool not found in tool registry")
```

This means the hotel finder agent no longer needs any manual `register()` calls. Instead of this:

```python
# Old way — manual, verbose, error-prone
action_registry = ActionRegistry()
action_registry.register(Action(name="geocode_city", function=geocode_city, ...))
action_registry.register(Action(name="search_hotels", function=search_hotels, ...))
action_registry.register(Action(name="terminate", function=terminate, ...))
```

You write this:

```python
# New way — declarative, automatic, tag-driven
action_registry = PythonActionRegistry(tags=["geocode", "search", "hotels", "system"])
```

The registry scans all registered tools, matches by tag, and loads exactly what the agent needs.

---

## What Automatic Inference Gives You

Let's make the benefit concrete. Say you want to add star-rating filtering to `search_hotels`:

**Before decorators** — three things to update:

```python
# 1. The function
def search_hotels(lat: float, lon: float, radius_km: float = 5, min_stars: int = None):
    ...

# 2. The parameters schema (separately)
"properties": {
    "lat":       {"type": "number"},
    "lon":       {"type": "number"},
    "radius_km": {"type": "number"},
    "min_stars": {"type": "integer"},   # ← must remember to add this
}

# 3. The description (separately)
description="Searches hotels... supports star filter..."  # ← must remember to update
```

**After decorators** — one thing to update:

```python
@register_tool(tags=["search", "hotels"])
def search_hotels(lat: float, lon: float, radius_km: float = 5, min_stars: int = None) -> List[Dict]:
    """Fetch hotels from OpenStreetMap via Overpass API.

    ...
    Args:
        lat: Latitude from geocode_city
        lon: Longitude from geocode_city
        radius_km: Search radius in kilometres (default 5)
        min_stars: Minimum star rating to include (optional)    # ← add here
    """
    # filter by min_stars if provided
    ...
```

The decorator automatically detects the new `min_stars` parameter, adds it to the schema as optional (because it has a default of `None`), and the updated docstring becomes the new description. Zero registration code touched.

---

## Why the Decorator Pattern Matters

Let's be direct about what changed and why it matters.

**Single source of truth.** The function is now the authoritative source for its name, description, parameters, and registration status. There is no second place to look and no second place to update.

**Automatic schema inference.** The decorator reads type hints and default values to build the JSON schema automatically. `lat: float` becomes `{"type": "number"}` and gets added to `required`. `radius_km: float = 5` becomes optional. Add a parameter to the function — it appears in the schema. Rename it — the schema updates. No extra steps.

**Tag-based tool organization.** Tags let you define your tools independently of your agents. You can write fifty hotel-related tools and give each agent a focused subset by specifying which tags it needs. A read-only agent takes `["geocode", "search"]`. A full agent adds `["system"]`. The tools don't change — only the lens you view them through does.

---

## Source code
- [Hotel Finder Agent](https://github.com/divyeshrajpura4114/ai-learning/tree/main/agentic-ai/game-components/hotel-finder-agent)

---

## References
- [AI Agents and Agentic AI with Python & Generative AI](https://www.coursera.org/learn/ai-agents-python)
- [LiteLLM](https://www.litellm.ai)

