---
title: Building AI Agents with the GAME Framework: A Complete Guide
date: 2026-03-07 10:00:00 +0530
categories: [AgenticAI, LiteLLM]
tags: [AgenticAI, LiteLLM]
author: Divyesh Rajpura

---

# Building AI Agents with the GAME Framework: A Complete Guide

Designing an AI agent is part engineering, part philosophy. You're not just writing code — you're defining how a machine thinks, remembers, and acts. But most agent implementations end up as tangled webs of conditional logic, hardcoded prompts, and brittle function calls.

The **GAME framework** offers a cleaner path. By decomposing every agent into four well-defined components — **Goals, Actions, Memory, and Environment** — you get agents that are modular, testable, and built to evolve.

This post walks you through each component in depth, shows how they fit together in a reusable agent loop, and demonstrates the full framework with a concrete, working example: a **Hotel Finder Agent** that takes a city and user preferences, calls real free APIs to fetch live hotel data, and returns a ranked shortlist of the top 5 options.

---

## Why GAME?

Before diving into components, it's worth asking: why bother with a framework at all?

The answer is reuse and clarity. The core loop of an AI agent almost never changes — construct a prompt, call the LLM, execute an action, update memory, repeat. What *does* change between agents is everything else: what they're trying to accomplish, what tools they have, how they remember things, and how they interact with the world.

GAME separates the **stable** (the loop) from the **variable** (the components), so you can swap out one agent's personality without touching the plumbing.

---

## G — Goals

### What Are Goals?

Goals define *what* the agent is trying to accomplish and *how* it should approach the task. They are the agent's guiding philosophy — the instructions it consults when deciding what to do next.

In GAME, a `Goal` is a simple data class:

```python
@dataclass(frozen=True)
class Goal:
    priority: int
    name: str
    description: str
```

The `priority` field lets you sort or weight goals when building prompts. `name` provides a human-readable label, and `description` contains the actual guidance the agent will see.

### Goals Are More Than Objectives

One of the subtler insights in GAME is that "goal" is a broad term. Goals can express:

- **Objectives**: *"Find hotels matching the user's criteria"*
- **Behavioral rules**: *"Always rank by relevance to criteria, not just star rating"*
- **Reasoning examples**: *"When you encounter an API error, try adjusting the search radius before giving up"*
- **Constraints**: *"Return exactly 5 results — no more, no less"*

This flexibility means goals do double duty — they tell the agent what to aim for *and* how to think along the way.

### Example: Hotel Finder Goals

```python
goals = [
    Goal(
        priority=1,
        name="Geocode Location",
        description=(
            "Convert the user's city name into geographic coordinates "
            "using the geocode_city action before doing anything else."
        )
    ),
    Goal(
        priority=2,
        name="Find Hotels",
        description=(
            "Search for hotels near the geocoded coordinates using search_hotels. "
            "Use a radius of 3-5km for city centres, larger for rural areas."
        )
    ),
    Goal(
        priority=3,
        name="Rank and Recommend",
        description=(
            "Analyse the retrieved hotels against the user's criteria — location, "
            "price tier, pet friendliness, star rating, and any other preferences. "
            "Call terminate with a ranked list of exactly the top 5 matches and a "
            "brief explanation for each recommendation."
        )
    ),
]
```

Notice how the goals encode not just *what* to do but *in what order* and *how to decide*. The third goal instructs the agent on the evaluation logic, saving you from having to hard-code that reasoning anywhere else.

---

## A — Actions

### What Are Actions?

Actions define what the agent *can do*. Think of them as the agent's toolkit — the discrete capabilities it can invoke to interact with the world.

Each action wraps a Python function along with the metadata needed for the LLM to understand and invoke it:

```python
class Action:
    def __init__(self,
                 name: str,
                 function: Callable,
                 description: str,
                 parameters: Dict,
                 terminal: bool = False):
        self.name = name
        self.function = function
        self.description = description
        self.terminal = terminal
        self.parameters = parameters

    def execute(self, **args) -> Any:
        return self.function(**args)
```

The `parameters` field uses a JSON Schema-style structure, which maps cleanly onto function calling APIs. The `terminal` flag signals to the agent loop that calling this action means the task is complete.

### The ActionRegistry

Actions are organized in a registry, which serves as a lookup table by name:

```python
class ActionRegistry:
    def __init__(self):
        self.actions = {}

    def register(self, action: Action):
        self.actions[action.name] = action

    def get_action(self, name: str) -> Action:
        return self.actions.get(name, None)

    def get_actions(self) -> List[Action]:
        return list(self.actions.values())
```

When the LLM returns a response specifying `"tool": "search_hotels"`, the registry is what maps that string back to the actual Python function.

### Defining Actions Well

The quality of your action descriptions directly impacts agent behavior. Compare these two definitions for the same function:

**Vague:**
```python
Action(
    name="search_hotels",
    description="Search for hotels.",
    ...
)
```

**Precise:**
```python
Action(
    name="search_hotels",
    description=(
        "Searches for hotels near a lat/lon coordinate using the Overpass API "
        "(OpenStreetMap data). Call geocode_city first to get coordinates. "
        "Returns a list of hotels with name, star rating, pet policy, "
        "address, and type (hotel/motel/guest_house). "
        "radius_km controls the search area — use 3 for dense cities, "
        "10+ for rural areas."
    ),
    ...
)
```

The second version tells the agent what data it will get back, when to call it, and how to tune the parameters. That context dramatically improves decision-making without changing a single line of the core loop.

### Hotel Finder Actions

Our hotel finder needs three actions: one to resolve a city name into coordinates, one to fetch hotels from OpenStreetMap, and one terminal action to deliver the final answer. Both data APIs are completely free and require no API key.

```python
import requests
from typing import List, Dict

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL  = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "GAME-HotelAgent/1.0", "Accept-Language": "en"}


def geocode_city(city: str) -> Dict:
    """Resolve a city name to lat/lon via Nominatim (free, no API key)."""
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
    return {
        "city": r["display_name"],
        "lat": float(r["lat"]),
        "lon": float(r["lon"]),
    }


def search_hotels(lat: float, lon: float, radius_km: float = 5) -> List[Dict]:
    """Fetch hotels from OpenStreetMap via Overpass API (free, no API key)."""
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
            "name":       tags.get("name"),
            "type":       tags.get("tourism", "hotel"),
            "stars":      tags.get("stars") or tags.get("stars:official"),
            "pets":       tags.get("pets") or tags.get("dog") or tags.get("pets_allowed"),
            "wifi":       tags.get("internet_access"),
            "wheelchair": tags.get("wheelchair"),
            "website":    tags.get("website") or tags.get("contact:website"),
            "address":    " ".join(filter(None, [
                              tags.get("addr:housenumber"),
                              tags.get("addr:street"),
                              tags.get("addr:city"),
                          ])),
        })
    return hotels


def terminate(message: str) -> str:
    """Deliver the final hotel recommendations to the user."""
    return message
```

Now register all three with the `ActionRegistry`:

```python
action_registry = ActionRegistry()

action_registry.register(Action(
    name="geocode_city",
    function=geocode_city,
    description=(
        "Converts a city name into geographic coordinates (lat/lon) "
        "using the Nominatim API. Always call this before search_hotels."
    ),
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, e.g. 'Paris' or 'Tokyo, Japan'"
            }
        },
        "required": ["city"]
    },
))

action_registry.register(Action(
    name="search_hotels",
    function=search_hotels,
    description=(
        "Searches for hotels near a lat/lon coordinate using the Overpass API "
        "(OpenStreetMap data). Returns name, star rating, pet policy, wifi, "
        "wheelchair access, address, and type for each hotel found. "
        "Use radius_km=3 for dense cities, 8-10 for suburban or rural areas."
    ),
    parameters={
        "type": "object",
        "properties": {
            "lat":       {"type": "number", "description": "Latitude from geocode_city"},
            "lon":       {"type": "number", "description": "Longitude from geocode_city"},
            "radius_km": {"type": "number", "description": "Search radius in km (default 5)"},
        },
        "required": ["lat", "lon"]
    },
))

action_registry.register(Action(
    name="terminate",
    function=terminate,
    description=(
        "Ends the session and delivers the final answer to the user. "
        "Call this after ranking hotels. The message should contain a "
        "numbered list of the top 5 hotels with a brief explanation for "
        "each, covering how well it matches the user's criteria."
    ),
    parameters={
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The final hotel recommendations"
            }
        },
        "required": ["message"]
    },
    terminal=True,
))
```

### The Terminal Action Pattern

Every well-designed agent needs at least one terminal action. For the hotel finder, `terminate` is the only way out of the loop — it forces the agent to commit to a final ranked answer rather than querying APIs indefinitely. The `terminal=True` flag is what the agent loop checks to know when to stop.

---

## M — Memory

### What Is Memory?

Memory is how the agent maintains context across loop iterations. Without it, every step would be isolated — the agent would forget the coordinates it just geocoded before it could search for hotels.

The base implementation is intentionally simple:

```python
class Memory:
    def __init__(self):
        self.items = []

    def add_memory(self, memory: dict):
        self.items.append(memory)

    def get_memories(self, limit: int = None) -> List[Dict]:
        return self.items[:limit]
```

Memories are stored as message-like dictionaries with a `type` (user or assistant) and `content`. This maps directly to the conversation format that LLMs expect.

### How Memory Flows Through a Hotel Search

In a typical hotel finder run, memory accumulates like this:

| Iteration | What gets stored |
|---|---|
| 0 | User request: *"Find pet-friendly hotels in Amsterdam under €150"* |
| 1 | Agent's `geocode_city("Amsterdam")` call + `{lat: 52.37, lon: 4.89}` returned |
| 2 | Agent's `search_hotels(52.37, 4.89, radius_km=4)` call + 23 hotels returned |
| 3 | Agent's `terminate(ranked_list)` call + final message delivered |

By iteration 3, the agent has the full context of what it searched and what it found, enabling it to write a well-reasoned recommendation that references specific hotels by name.

### Why Wrap a Simple List?

It might seem like overkill to wrap a list in a class. But the abstraction pays off quickly. Because memory is accessed through an interface, you can later swap it for:

- A **database-backed** implementation that persists searches across sessions
- A **summarizing** implementation that compresses old hotel lists to save tokens
- A **caching** implementation that skips the Overpass call if the same city was searched recently

None of these changes require touching the agent loop. The loop just calls `get_memories()` — it doesn't care what's happening underneath.

---

## E — Environment

### What Is the Environment?

The Environment is the bridge between the agent and the real world. It executes actions and returns results, wrapping every call in consistent error handling and metadata.

```python
class Environment:
    def execute_action(self, action: Action, args: dict) -> dict:
        try:
            result = action.execute(**args)
            return self.format_result(result)
        except Exception as e:
            return {
                "tool_executed": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def format_result(self, result: Any) -> dict:
        return {
            "tool_executed": True,
            "result": result,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")
        }
```

For the hotel finder, the default `Environment` works perfectly — our actions are straightforward HTTP calls to Nominatim and Overpass. But the abstraction matters. You could substitute:

- A **MockEnvironment** that returns fixture data for testing without hitting real APIs
- A **RateLimitedEnvironment** that enforces pauses between Overpass calls
- A **CachedEnvironment** that stores Overpass responses to avoid duplicate network requests

The agent loop calls `environment.execute_action(action, args)` and never needs to know which environment it's running in.

---

## The Agent Language: The Missing Piece

GAME's four components cover *what* the agent knows and can do. But there's a fifth concern that cuts across all of them: *how does the agent communicate with the LLM?*

This is the role of `AgentLanguage`.

### Two Responsibilities

`AgentLanguage` has two jobs:

1. **Prompt Construction** — Transform Goals, Actions, and Memory into a prompt the LLM can understand
2. **Response Parsing** — Interpret the LLM's response to determine which action was chosen

```python
class AgentLanguage:
    def construct_prompt(self, actions, environment, goals, memory) -> Prompt:
        raise NotImplementedError

    def parse_response(self, response: str) -> dict:
        raise NotImplementedError
```

### Function Calling Language

For the hotel finder we use `AgentFunctionCallingActionLanguage`. The LLM uses native function calling to return structured actions directly — no fragile text parsing required:

```python
class AgentFunctionCallingActionLanguage(AgentLanguage):

    def format_actions(self, actions: List[Action]) -> List:
        return [
            {
                "type": "function",
                "function": {
                    "name": action.name,
                    "description": action.description[:1024],
                    "parameters": action.parameters,
                },
            }
            for action in actions
        ]

    def construct_prompt(self, actions, environment, goals, memory) -> Prompt:
        prompt = []
        prompt += self.format_goals(goals)
        prompt += self.format_memory(memory)
        tools = self.format_actions(actions)
        return Prompt(messages=prompt, tools=tools)

    def parse_response(self, response: str) -> dict:
        try:
            return json.loads(response)
        except Exception:
            # Fall back to terminate if the response can't be parsed
            return {"tool": "terminate", "args": {"message": response}}
```

Being able to swap `AgentLanguage` implementations means you can experiment with different prompt strategies or support different LLM providers without touching anything else in the agent.

---

## The Agent Loop

All five components come together in the `Agent` class and its `run` method:

```python
def run(self, user_input: str, memory=None, max_iterations: int = 10) -> Memory:
    memory = memory or Memory()
    self.set_current_task(memory, user_input)

    for _ in range(max_iterations):
        # 1. Construct prompt from Goals, Actions, Memory
        prompt = self.construct_prompt(self.goals, memory, self.actions)

        # 2. Ask the LLM what to do next
        response = self.prompt_llm_for_action(prompt)
        print(f"Agent Decision: {response}")

        # 3. Parse the response to identify the chosen action
        action, invocation = self.get_action(response)

        # 4. Execute the action in the Environment (hits real APIs)
        result = self.environment.execute_action(action, invocation["args"])
        print(f"Action Result: {result}")

        # 5. Store the decision and result in Memory
        self.update_memory(memory, response, result)

        # 6. Check if the agent called a terminal action
        if self.should_terminate(response):
            break

    return memory
```

Six steps, every iteration, forever — until the agent terminates or hits the maximum iteration limit. The elegance here is that *none of this code changes* between agents. Only the GAME components change.

For the hotel finder, `max_iterations=10` is generous — a well-designed agent completes the task in exactly 3 steps: geocode → search → terminate.

---

## Putting It All Together: Complete Hotel Finder Implementation

```python
import os
import json
import time
import traceback
import requests
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import anthropic


# ── Core GAME classes ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Goal:
    priority: int
    name: str
    description: str


class Action:
    def __init__(self, name, function, description, parameters, terminal=False):
        self.name = name
        self.function = function
        self.description = description
        self.parameters = parameters
        self.terminal = terminal

    def execute(self, **args):
        return self.function(**args)


class ActionRegistry:
    def __init__(self):
        self.actions = {}

    def register(self, action: Action):
        self.actions[action.name] = action

    def get_action(self, name: str) -> Optional[Action]:
        return self.actions.get(name)

    def get_actions(self) -> List[Action]:
        return list(self.actions.values())


class Memory:
    def __init__(self):
        self.items = []

    def add_memory(self, memory: dict):
        self.items.append(memory)

    def get_memories(self, limit: int = None) -> List[Dict]:
        return self.items[:limit]


class Environment:
    def execute_action(self, action: Action, args: dict) -> dict:
        try:
            result = action.execute(**args)
            return {"tool_executed": True, "result": result,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        except Exception as e:
            return {"tool_executed": False, "error": str(e),
                    "traceback": traceback.format_exc()}


# ── Hotel-specific action functions ─────────────────────────────────────────

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL  = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "GAME-HotelAgent/1.0", "Accept-Language": "en"}


def geocode_city(city: str) -> Dict:
    resp = requests.get(
        NOMINATIM_URL,
        params={"q": city, "format": "json", "limit": 1},
        headers=HEADERS, timeout=10,
    )
    results = resp.json()
    if not results:
        return {"error": f"City '{city}' not found"}
    r = results[0]
    return {"city": r["display_name"], "lat": float(r["lat"]), "lon": float(r["lon"])}


def search_hotels(lat: float, lon: float, radius_km: float = 5) -> List[Dict]:
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
            "name":       tags.get("name"),
            "type":       tags.get("tourism", "hotel"),
            "stars":      tags.get("stars") or tags.get("stars:official"),
            "pets":       tags.get("pets") or tags.get("dog") or tags.get("pets_allowed"),
            "wifi":       tags.get("internet_access"),
            "wheelchair": tags.get("wheelchair"),
            "website":    tags.get("website") or tags.get("contact:website"),
            "address":    " ".join(filter(None, [
                              tags.get("addr:housenumber"),
                              tags.get("addr:street"),
                              tags.get("addr:city"),
                          ])),
        })
    return hotels


def terminate(message: str) -> str:
    return message


# ── Agent class ──────────────────────────────────────────────────────────────

class Agent:
    def __init__(self, goals, action_registry, generate_response, environment):
        self.goals = goals
        self.actions = action_registry
        self.generate_response = generate_response
        self.environment = environment

    def _build_system_prompt(self) -> str:
        goal_text = "\n".join(
            f"{g.priority}. {g.name}: {g.description}"
            for g in sorted(self.goals, key=lambda g: g.priority)
        )
        return f"You are a hotel-finding agent. Follow these goals in order:\n\n{goal_text}"

    def _get_tools(self) -> List[Dict]:
        return [
            {
                "name": a.name,
                "description": a.description,
                "input_schema": a.parameters if a.parameters else {
                    "type": "object", "properties": {}, "required": []
                },
            }
            for a in self.actions.get_actions()
        ]

    def run(self, user_input: str, max_iterations: int = 10) -> str:
        memory = Memory()
        memory.add_memory({"role": "user", "content": user_input})

        for _ in range(max_iterations):
            # Build messages from memory
            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in memory.get_memories()
            ]

            # Call the LLM
            response = self.generate_response(
                system=self._build_system_prompt(),
                messages=messages,
                tools=self._get_tools(),
            )

            # Check for tool use
            tool_use_block = next(
                (b for b in response.content if b.type == "tool_use"), None
            )

            if tool_use_block is None:
                # No tool call — extract text and terminate
                text = next(
                    (b.text for b in response.content if hasattr(b, "text")), ""
                )
                return text

            tool_name = tool_use_block.name
            tool_args = tool_use_block.input
            tool_use_id = tool_use_block.id

            print(f"\nAgent calls: {tool_name}({tool_args})")

            # Look up and execute the action
            action = self.actions.get_action(tool_name)
            result = self.environment.execute_action(action, tool_args)

            print(f"Result: {json.dumps(result['result'], indent=2)[:300]}...")

            # Store the exchange in memory
            memory.add_memory({
                "role": "assistant",
                "content": response.content,
            })
            memory.add_memory({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": json.dumps(result["result"]),
                }],
            })

            # Stop if this was a terminal action
            if action.terminal:
                return result["result"]

        return "Max iterations reached without a final answer."


# ── Wiring it up ─────────────────────────────────────────────────────────────

def main():
    goals = [
        Goal(
            priority=1,
            name="Geocode Location",
            description=(
                "Convert the user's city name into coordinates "
                "using geocode_city before doing anything else."
            )
        ),
        Goal(
            priority=2,
            name="Find Hotels",
            description=(
                "Search for hotels near the coordinates using search_hotels. "
                "Use radius_km=3-5 for cities, larger for rural areas."
            )
        ),
        Goal(
            priority=3,
            name="Rank and Recommend",
            description=(
                "Analyse hotels against the user's criteria: location, price tier "
                "(1-2 stars=budget, 3=mid-range, 4-5=luxury), pet friendliness, "
                "wifi, and any other stated preferences. "
                "Call terminate with a ranked list of exactly the top 5 matches, "
                "with a brief explanation for each covering why it fits the criteria."
            )
        ),
    ]

    registry = ActionRegistry()

    registry.register(Action(
        name="geocode_city",
        function=geocode_city,
        description=(
            "Converts a city name into geographic coordinates using the Nominatim API. "
            "Always call this before search_hotels."
        ),
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Amsterdam'"}
            },
            "required": ["city"]
        },
    ))

    registry.register(Action(
        name="search_hotels",
        function=search_hotels,
        description=(
            "Fetches hotels from OpenStreetMap via the Overpass API. "
            "Returns name, stars, pets policy, wifi, address, and type. "
            "Use radius_km=3 for dense cities, up to 10 for rural areas."
        ),
        parameters={
            "type": "object",
            "properties": {
                "lat":       {"type": "number"},
                "lon":       {"type": "number"},
                "radius_km": {"type": "number", "description": "Search radius (default 5)"},
            },
            "required": ["lat", "lon"]
        },
    ))

    registry.register(Action(
        name="terminate",
        function=terminate,
        description=(
            "Ends the session with the final hotel recommendations. "
            "Include a numbered top-5 list with a short explanation per hotel."
        ),
        parameters={
            "type": "object",
            "properties": {
                "message": {"type": "string"}
            },
            "required": ["message"]
        },
        terminal=True,
    ))

    client = anthropic.Anthropic()

    def generate_response(system, messages, tools):
        return client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            system=system,
            messages=messages,
            tools=tools,
        )

    agent = Agent(
        goals=goals,
        action_registry=registry,
        generate_response=generate_response,
        environment=Environment(),
    )

    user_input = input("Describe what you're looking for (city + preferences):\n> ")
    # Example: "Find pet-friendly budget hotels in Amsterdam"

    print("\n── Agent Running ──────────────────────────────────────")
    answer = agent.run(user_input, max_iterations=10)

    print("\n── Hotel Recommendations ──────────────────────────────")
    print(answer)


if __name__ == "__main__":
    main()
```

### What Happens at Runtime

A typical run for *"pet-friendly mid-range hotels in Amsterdam"* completes in exactly three iterations:

**Iteration 1 — Geocode**

The agent calls `geocode_city("Amsterdam")` via Nominatim and receives coordinates. These get stored in memory.

**Iteration 2 — Search**

Using the coordinates from memory, the agent calls `search_hotels(52.3676, 4.9041, radius_km=4)`. The Overpass API returns real hotels with their OpenStreetMap tags — star ratings, pets policies, wifi status, addresses.

**Iteration 3 — Rank and Terminate**

The agent analyses the hotel list against the criteria. It infers price tier from star ratings, checks pets tags, and calls `terminate` with a ranked shortlist like:

```
1. Hotel V Nesplein ★★★★ — Pet-friendly (confirmed in OSM data),
   mid-range price, central location near Vondelpark. Good wifi.

2. Conscious Hotel Vondelpark ★★★ — Pets allowed, eco-certified,
   walking distance to Vondelpark. Strong value for money.

3. The Flying Pig Downtown ★★ — Budget-friendly, pet policy confirmed,
   great for couples. Central location near Leidseplein.

4. Hotel Hegra ★★★ — Small boutique hotel, no explicit pet policy
   listed but guest_house type suggests flexibility. Worth calling ahead.

5. Stayokay Amsterdam Vondelpark ★★ — Budget option with pet-friendly
   rooms confirmed. Books fast in peak season.
```

---

## Simulate Before You Build

One of the most valuable techniques when designing a GAME agent is *simulation*. Before writing any code, you can test your agent design by prompting an LLM in a chat interface to roleplay as your agent.

Here is the simulation prompt for the hotel finder:

```
I'd like to simulate an AI agent I'm designing.

Goals:
- Convert city name to coordinates using geocode_city
- Search for hotels using search_hotels
- Rank by user criteria and return the top 5

Actions available:
- geocode_city(city) → {lat, lon, city}
- search_hotels(lat, lon, radius_km) → [{name, stars, pets, wifi, address}, ...]
- terminate(message) → delivers final recommendations

At each step, output one action to take.
Stop and wait — I'll type the result as my next message.

Ask me for the first task.
```

Then try scenarios like:

- Returning `{"error": "City not found"}` from `geocode_city` to see if the agent tries alternate spellings
- Returning an empty hotel list to see if it widens the radius before giving up
- Returning hotels with no star ratings to see how it handles missing price data

Simulation catches design flaws early. If the simulated agent can't reason well about missing pet policy data, your goal descriptions need to be more explicit about how to handle that case — before you've written a single line of real code.

---

## The Power of Modularity

The true payoff of GAME becomes clear when you realize how easily you can create entirely different agents by swapping components. The hotel finder's loop is identical to what you'd use for a restaurant finder, a flight searcher, or a real estate scout:

```python
# Restaurant finder — swap goals and actions, keep everything else
restaurant_agent = Agent(
    goals=[Goal(1, "Find Restaurants", "Find top 5 restaurants matching cuisine and budget")],
    action_registry=ActionRegistry([GeocodeCityAction(), SearchRestaurantsAction(), TerminateAction()]),
    generate_response=generate_response,
    environment=Environment()
)

# Flight searcher — different APIs, same loop
flight_agent = Agent(
    goals=[Goal(1, "Find Flights", "Find cheapest flights matching dates and budget")],
    action_registry=ActionRegistry([ParseDatesAction(), SearchFlightsAction(), TerminateAction()]),
    generate_response=generate_response,
    environment=Environment()
)
```

Same loop. Completely different behavior. This is the promise of GAME: the work of building a new agent is defining what it should do — not rewiring how agents work.

---

## Key Takeaways

| Component | Role | Hotel Finder Example |
|---|---|---|
| **Goals** | Define what the agent wants and how it should think | Geocode → Search → Rank by criteria → Top 5 only |
| **Actions** | Define what the agent can do | `geocode_city` (Nominatim), `search_hotels` (Overpass), `terminate` |
| **Memory** | Maintain context across iterations | Coordinates from geocode feed directly into hotel search |
| **Environment** | Execute actions and return results | Wraps all HTTP calls, handles errors uniformly |
| **AgentLanguage** | Translate between GAME components and LLM I/O | Function calling for reliable structured output |

---

## References
[AI Agents and Agentic AI with Python & Generative AI](https://www.coursera.org/learn/ai-agents-python)
[LiteLLM](https://www.litellm.ai)