# Interface Prompt Summary

Implemented in `code/agents/interfaces.py`.

## Raw JSON

The model must emit exactly one JSON action:

```json
{"kind":"tool","tool_name":"search_availability","tool_args":{"attendee":"Ari","date":"2026-06-01","time":"09:00","timezone":"UTC"}}
```

Invalid JSON is counted as a parse failure.

## JSON + ParseRepair

The first prompt is the same as Raw JSON. If parsing fails, the model receives one syntax-only retry prompt. The retry is not allowed to change task logic; it only repairs malformed JSON.

## Schema-Guided

The prompt repeats available tools, required arguments, current state summary, required effects, and a reminder not to finalize until required effects are complete.

## Tool-Call Compiler

The model may emit a natural-language intent plus arguments. A deterministic adapter maps the intent to the closest admissible tool call. The adapter uses only task metadata and visible observations. If required fields are missing, it emits a clarification action rather than inventing facts.

This mode is used to reduce tool syntax brittleness and expose mid-trajectory state failures in local models.
