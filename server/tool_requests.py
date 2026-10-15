"""Normalize OpenAI tool history before the checkpoint template sees it."""
import re
from .api import ApiError, text_content
from .vision import decode_url, image_url_of
from .json_values import history_arguments, json_value
from .tool_schema import check_schema, WorkBudget

NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")

def function_name(value, field):
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise ApiError("Function names must use 1 to 64 letters, digits, underscores, or hyphens.", field)
    return value

def validate_tools(value):
    if not isinstance(value, list) or len(value) > 128:
        raise ApiError("tools must be a list of at most 128 functions.", "tools")
    normalized, names = [], set()
    budget = WorkBudget()
    for item in value:
        if not isinstance(item, dict) or set(item) - {"type", "function"} or item.get("type") != "function":
            raise ApiError("Each tool must have type function and a function object.", "tools")
        function = item.get("function")
        if not isinstance(function, dict) or set(function) - {"name", "description", "parameters", "strict"}:
            raise ApiError("A function supports name, description, parameters, and strict only.", "tools")
        name = function_name(function.get("name"), "tools")
        if name in names:
            raise ApiError("Tool function names must be unique.", "tools")
        names.add(name)
        if "description" in function and not isinstance(function["description"], str):
            raise ApiError("A tool description must be text.", "tools")
        if "strict" in function and type(function["strict"]) is not bool:
            raise ApiError("A tool strict flag must be true or false.", "tools")
        parameters = function.get("parameters", dict(type="object", properties={}))
        if not isinstance(parameters, dict) or parameters.get("type", "object") != "object":
            raise ApiError("Function parameters must describe a JSON object.", "tools")
        properties = parameters.get("properties", {})
        if not isinstance(properties, dict):
            raise ApiError("Function parameter properties must be an object.", "tools")
        for key, schema in properties.items():
            if (not isinstance(key, str) or not key or len(key) > 256 or any(char in key for char in "<>\r\n") or
                    not isinstance(schema, (dict, bool))):
                raise ApiError("Parameter names must fit a template tag; each schema must be an object or boolean.", "tools")
        try:
            budget.inspect(parameters)
            tool = json_value(dict(type="function", function=function | dict(parameters=parameters)), "tools")
            check_schema(tool["function"]["parameters"], tool["function"]["parameters"], budget=budget)
        except (ValueError, TypeError, RecursionError):
            raise ApiError("The tool schema is invalid or exceeds its size, depth, reference, or work limit.", "tools") from None
        normalized.append(tool)
    return tuple(normalized)

def validate_choice(value, tools):
    if isinstance(value, str) and value in ("auto", "none", "required"):
        if value == "required" and not tools:
            raise ApiError("tool_choice required needs at least one tool.", "tool_choice")
        return value
    if (isinstance(value, dict) and set(value) == {"type", "function"} and value.get("type") == "function" and
            isinstance(value.get("function"), dict) and set(value["function"]) == {"name"}):
        name = function_name(value["function"]["name"], "tool_choice")
        if any(tool["function"]["name"] == name for tool in tools):
            return dict(type="function", function=dict(name=name))
    raise ApiError("tool_choice must be auto, none, required, or one advertised function.", "tool_choice")

def history_calls(value):
    try:
        return checked_history_calls(value)
    except ApiError as error:
        raise ApiError(str(error), error.param, error.status, code="invalid_tool_call_history") from None

def checked_history_calls(value):
    if not isinstance(value, list) or not value or len(value) > 128:
        raise ApiError("Assistant tool_calls must be a nonempty list of at most 128 calls.", "messages")
    result = []
    for call in value:
        if (not isinstance(call, dict) or set(call) != {"id", "type", "function"} or call.get("type") != "function" or
                not isinstance(call.get("id"), str) or not 0 < len(call["id"]) <= 256):
            raise ApiError("Each assistant tool call needs id, type function, and a function object.", "messages")
        function = call["function"]
        if not isinstance(function, dict) or set(function) != {"name", "arguments"}:
            raise ApiError("Each tool call function needs name and arguments.", "messages")
        result.append(dict(id=json_value(call["id"], "messages"), type="function", function=dict(
            name=function_name(function["name"], "messages"), arguments=history_arguments(function["arguments"], "messages"))))
    return result

def message_content(content, role, vision, images):
    """Text content as before, or a part list with image placeholders when the request carries images."""
    if not isinstance(content, list):
        return text_content(content)
    urls = [image_url_of(part) if isinstance(part, dict) else None for part in content]
    if all(url is None for url in urls):
        return text_content(content)
    if vision is None:
        raise ApiError("This server runs without image input. Start it with --vision to send images.", "messages")
    if role not in ("user", "tool"):
        raise ApiError("Images are supported in user and tool messages only.", "messages")
    parts = []
    for part, url in zip(content, urls):
        if url is None:
            parts.append(dict(type="text", text=text_content([part])))
            continue
        if len(images) >= vision.max_images:
            raise ApiError(f"A request can contain at most {vision.max_images} images.", "messages")
        images.append(decode_url(url, vision, sum(image.tokens for image in images)))
        parts.append(dict(type="image"))
    return parts

def normalize_message(item, index, vision=None, images=None):
    if not isinstance(item, dict):
        raise ApiError("Each message must be an object.", "messages")
    role = item.get("role")
    allowed = {"role", "content", "reasoning_content"}
    if role == "assistant":
        allowed |= {"tool_calls"}
    if role == "tool":
        allowed |= {"tool_call_id", "name"}
    if set(item) - allowed:
        raise ApiError("The message contains unsupported fields for its role.", "messages")
    if role not in ("system", "user", "assistant", "tool") or (role == "system" and index != 0):
        raise ApiError("Use system first, then user, assistant, or tool messages.", "messages")
    reasoning = item.get("reasoning_content", "")
    if not isinstance(reasoning, str):
        raise ApiError("reasoning_content must be a string.", "messages")
    calls = history_calls(item["tool_calls"]) if "tool_calls" in item else None
    content = ("" if role == "assistant" and calls and item.get("content") is None
               else message_content(item.get("content"), role, vision, images if images is not None else []))
    result = dict(role=role, content=json_value(content, "messages"), reasoning_content=json_value(reasoning, "messages"))
    if calls:
        result["tool_calls"] = calls
    if role == "tool":
        if not isinstance(item.get("tool_call_id"), str) or not item["tool_call_id"]:
            raise ApiError("A tool message needs a tool_call_id.", "messages")
        result["tool_call_id"] = json_value(item["tool_call_id"], "messages")
        if "name" in item:
            result["name"] = function_name(item["name"], "messages")
    return result

def chat_messages(value, vision=None, images=None):
    if not isinstance(value, list) or not value:
        raise ApiError("messages must be a nonempty list.", "messages")
    messages, pending, seen = [], {}, set()
    for index, item in enumerate(value):
        message = normalize_message(item, index, vision, images)
        if message["role"] == "tool":
            call_id = message["tool_call_id"]
            if call_id not in pending or ("name" in message and message["name"] != pending[call_id]):
                raise ApiError("A tool result must match one pending assistant tool call.", "messages")
            pending = {key: name for key, name in pending.items() if key != call_id}
        elif pending:
            raise ApiError("Return all pending tool results before the next conversation message.", "messages")
        for call in message.get("tool_calls", []):
            if call["id"] in seen:
                raise ApiError("Tool call IDs must be unique in the conversation.", "messages")
            seen.add(call["id"])
            pending = pending | {call["id"]: call["function"]["name"]}
        messages.append(message)
    if pending:
        raise ApiError("The conversation has tool calls without tool results.", "messages")
    if not any(item["role"] == "user" for item in messages):
        raise ApiError("messages must contain a user query.", "messages")
    return tuple(messages)
