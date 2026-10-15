import copy
import json
import random
import unittest
from server.api import ApiError, validate_request
from server.tool_output import ToolOutput
from server.tool_schema import check_schema, matches, types
from server.tests import test_http


def reference_dag(keyword, depth=20):
    definitions = {"d0": dict(type="integer")}
    for level in range(1, depth + 1):
        definitions["d" + str(level)] = {keyword: [{"$ref": "#/$defs/d" + str(level - 1)} for _ in range(4)]}
    return dict(type="object", properties=dict(value={"$ref": "#/$defs/d" + str(depth)}), **{"$defs": definitions})


def request_body(schema):
    return dict(messages=[dict(role="user", content="Call f")], tools=[dict(type="function", function=dict(
        name="f", parameters=schema))])


class SchemaWorkTests(unittest.TestCase):
    def test_review_reference_dags_are_linear(self):
        for keyword in ("allOf", "anyOf", "oneOf"):
            root = reference_dag(keyword)
            validate_request(request_body(root), "chat", "crucible")
            self.assertEqual(types(root["properties"]["value"], root), {"integer"} if keyword != "allOf" else set())
            self.assertEqual(matches(1, root["properties"]["value"], root), keyword != "oneOf")
            self.assertFalse(matches("wrong", root["properties"]["value"], root))

    def test_limits_rejected_before_submit(self):
        for schema in (
            dict(type="object", description="x" * 131073),
            dict(type="object", properties={"p" + str(i): True for i in range(5000)}),
            dict(type="object", **{"$defs": {"d": True}}, allOf=[{"$ref": "#/$defs/d"} for _ in range(513)]),
            dict(type="object", **{"$defs": {"loop": {"$ref": "#/$defs/loop"}}}, allOf=[{"$ref": "#/$defs/loop"}]),
        ):
            with self.assertRaises(ApiError) as raised:
                validate_request(request_body(schema), "chat", "crucible")
            self.assertEqual(raised.exception.status, 400)
            self.assertIn("schema", str(raised.exception).lower())

    def test_depth_and_reference_chain_limits(self):
        deep = True
        for _ in range(65):
            deep = dict(items=deep)
        root = dict(type="object", properties=dict(value=deep))
        with self.assertRaises(ApiError) as raised:
            validate_request(request_body(root), "chat", "crucible")
        self.assertEqual(raised.exception.status, 400)
        definitions = {"d0": dict(type="integer")}
        for i in range(1, 34):
            definitions["d" + str(i)] = {"$ref": "#/$defs/d" + str(i - 1)}
        root = dict(type="object", properties=dict(value={"$ref": "#/$defs/d33"}), **{"$defs": definitions})
        with self.assertRaisesRegex(ValueError, "reference chain"):
            types(root["properties"]["value"], root)

    def test_unused_definitions_are_validated(self):
        root = dict(type="object", **{"$defs": dict(unused=dict(type="invalid"))})
        with self.assertRaises(ApiError):
            validate_request(request_body(root), "chat", "crucible")

    def test_input_budget_is_shared_across_functions(self):
        schema = dict(type="object", properties={"p" + str(index): dict(type="integer") for index in range(1000)})
        body = request_body(schema)
        body["tools"] = [dict(type="function", function=dict(name="f" + str(i), parameters=schema)) for i in range(5)]
        with self.assertRaises(ApiError) as raised:
            validate_request(body, "chat", "crucible")
        self.assertEqual(raised.exception.status, 400)

    def test_matching_cache_uses_instance_path_and_call_scope(self):
        root = dict(type="object", properties=dict(a=dict(type="integer"), b=dict(type="integer")))
        shared = dict(type="integer")
        root["properties"] = dict(a=shared, b=shared)
        self.assertFalse(matches(dict(a=1, b="wrong"), root, root))
        body = request_body(root)
        output = ToolOutput(validate_request(body, "chat", "crucible"))
        good = "<tool_call><function=f><parameter=a>1</parameter><parameter=b>2</parameter></function></tool_call>"
        bad = good.replace("<parameter=a>1", "<parameter=a>\"wrong\"")
        output.feed(good + bad)
        message, _, reason = output.finish("stop")
        # The second call's mismatch is found (no cached match from the first call). A schema-only failure of a
        # complete call now goes out as a call with its reason recorded (design decision, Oct 8).
        self.assertEqual((reason, len(message["tool_calls"])), ("tool_calls", 2))
        self.assertEqual((output.fallback, output.unchecked), (False, ["schema_mismatch"]))

    def test_output_budget_is_shared_across_calls(self):
        schema = dict(type="object", properties=dict(values=dict(type="array", items=dict(type="integer"))))
        request = validate_request(request_body(schema), "chat", "crucible")
        block = ("<tool_call><function=f><parameter=values>" + json.dumps([0] * 1500) +
                 "</parameter></function></tool_call>")
        single = ToolOutput(request)
        single.feed(block)
        single.finish("stop")
        output = ToolOutput(request)
        output.feed(block + block)
        message, _, _ = output.finish("stop")
        self.assertNotIn("tool_calls", message)
        self.assertTrue(output.fallback)

    def test_model_matching_budget_returns_explicit_error(self):
        schema = dict(type="object", properties=dict(values=dict(type="array", items=dict(type="integer"))))
        output = ToolOutput(validate_request(request_body(schema), "chat", "crucible"))
        output.feed("<tool_call><function=f><parameter=values>" + json.dumps([0] * 20000) +
                    "</parameter></function></tool_call>")
        message, _, _ = output.finish("stop")
        self.assertNotIn("tool_calls", message)
        self.assertTrue(output.fallback)

    def test_http_limits_return_400_without_submission(self):
        for schema in (dict(type="object", description="x" * 131073),
                       dict(type="object", **{"$defs": {"loop": {"$ref": "#/$defs/loop"}}},
                            allOf=[{"$ref": "#/$defs/loop"}])):
            status, _, raw, backend = test_http.HttpTests().handle(request_body(schema))
            self.assertEqual(status, 400)
            self.assertEqual(backend.jobs, [])
            self.assertEqual(json.loads(raw)["error"]["param"], "tools")

    def test_fuzz_invalid_schemas_return_400(self):
        rng = random.Random(9094)
        invalid = [{"$ref": "external.json"}, {"$ref": "#/$defs/missing"}, dict(type=0), dict(type="wrong"),
                   dict(anyOf=[]), dict(allOf="wrong"), dict(required=[0]), dict(maxItems=-1), dict(properties=[])]
        for _ in range(200):
            root = reference_dag(rng.choice(["allOf", "anyOf", "oneOf"]), rng.randrange(1, 10))
            root["$defs"]["d0"] = copy.deepcopy(rng.choice(invalid))
            status, _, _, backend = test_http.HttpTests().handle(request_body(root))
            self.assertEqual(status, 400)
            self.assertEqual(backend.jobs, [])

    def test_fuzz_reference_dags_match_flattened_schema(self):
        rng = random.Random(70821)
        for _ in range(100):
            definitions = {"d0": rng.choice([dict(type="integer", minimum=0), dict(type="string", minLength=2),
                                             dict(enum=[0, "ok"]), dict(const=1)])}
            flattened = copy.deepcopy(definitions["d0"])
            for level in range(1, rng.randrange(2, 6)):
                keyword, width = rng.choice(["allOf", "anyOf", "oneOf"]), rng.randrange(1, 5)
                definitions["d" + str(level)] = {keyword: [{"$ref": "#/$defs/d" + str(level - 1)} for _ in range(width)]}
                flattened = {keyword: [copy.deepcopy(flattened) for _ in range(width)]}
            root = dict(type="object", properties=dict(value={"$ref": "#/$defs/d" + str(level)}), **{"$defs": definitions})
            check_schema(root, root)
            for value in (-1, 0, 1, True, "ok", "x", None, [], {}):
                self.assertEqual(matches(value, root["properties"]["value"], root), matches(value, flattened, flattened))


if __name__ == "__main__": unittest.main()
