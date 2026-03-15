"""Convert ToolACE dataset to Qwen3 native tool calling format."""

import ast
import json
import re


def extract_tools_from_system(system: str) -> list[dict] | None:
    """Extract JSON tool definitions from ToolACE system prompt."""
    try:
        start = system.index('[{')
    except ValueError:
        return None

    depth = 0
    for i in range(start, len(system)):
        if system[i] == '[':
            depth += 1
        elif system[i] == ']':
            depth -= 1
        if depth == 0:
            try:
                return json.loads(system[start:i + 1])
            except json.JSONDecodeError:
                return None
    return None


def toolace_to_openai_tools(tools: list[dict]) -> list[dict] | None:
    """Convert ToolACE tool defs to OpenAI function calling format."""
    result = []
    for tool in tools:
        if "name" not in tool:
            return None
        params = tool.get("parameters", {})
        # ToolACE uses "dict" instead of "object"
        if params.get("type") == "dict":
            params = {**params, "type": "object"}
        # Clean up properties
        properties = params.get("properties", {})
        cleaned_props = {}
        for k, v in properties.items():
            cleaned_props[k] = {
                "type": v.get("type", "string"),
                "description": v.get("description", ""),
            }
            if "enum" in v:
                cleaned_props[k]["enum"] = v["enum"]
            if "default" in v:
                cleaned_props[k]["default"] = v["default"]

        result.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", tool.get("desc", "")),
                "parameters": {
                    "type": "object",
                    "properties": cleaned_props,
                    "required": params.get("required", []),
                },
            },
        })
    return result


def parse_bracket_calls(text: str) -> list[dict]:
    """Parse '[FuncName(arg=val)]' into tool_calls format."""
    text = text.strip()
    # Remove outer brackets if present
    if text.startswith('[') and text.endswith(']'):
        text = text[1:-1]

    calls = []
    # Match function_name(args) patterns - name can have spaces
    for match in re.finditer(r'([\w][\w\s]*?)\(([^)]*)\)', text):
        name = match.group(1).strip()
        args_str = match.group(2).strip()

        kwargs = {}
        if args_str:
            try:
                tree = ast.parse(f'f({args_str})', mode='eval')
                for kw in tree.body.keywords:
                    try:
                        kwargs[kw.arg] = ast.literal_eval(kw.value)
                    except (ValueError, TypeError):
                        kwargs[kw.arg] = ast.unparse(kw.value)
            except SyntaxError:
                # Fallback: simple split
                for pair in args_str.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        v = v.strip().strip("\"'")
                        kwargs[k.strip()] = v

        calls.append({
            "type": "function",
            "function": {
                "name": name,
                "arguments": kwargs,
            },
        })
    return calls


def is_tool_call(text: str) -> bool:
    """Check if assistant response is a tool call (bracket notation)."""
    text = text.strip()
    return bool(re.match(r'\[.*\(.*\)\]', text, re.DOTALL))


ROLE_MAP = {"user": "user", "assistant": "assistant", "tool": "tool"}


def convert_toolace_example(example) -> dict | None:
    """Convert a single ToolACE example to Qwen3 tool calling format.

    Returns: {"messages": [...], "tools": [...]} or None if unparseable.
    """
    system = example.get("system", "")
    tools = extract_tools_from_system(system)
    if tools is None:
        return None

    openai_tools = toolace_to_openai_tools(tools)
    if openai_tools is None:
        return None

    messages = []
    for turn in example["conversations"]:
        role = ROLE_MAP.get(turn["from"], "user")
        content = turn["value"]

        if role == "assistant" and is_tool_call(content):
            # Convert bracket call to tool_calls format
            tool_calls = parse_bracket_calls(content)
            messages.append({
                "role": "assistant",
                "tool_calls": tool_calls,
            })
        elif role == "tool":
            messages.append({
                "role": "tool",
                "content": content,
            })
        else:
            messages.append({
                "role": role,
                "content": content,
            })

    return {"messages": messages, "tools": openai_tools}
