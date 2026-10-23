"""Host header rule: any Host by default, a strict list with CRUCIBLE_ALLOWED_HOSTS or --allowed-host."""
import json
import subprocess
import sys
import unittest
from types import SimpleNamespace

from server.hosts import ALLOWED_HOSTS_ENV, HostError, parse_allowed_hosts, resolve_allowed_hosts
from server.tests.serve_fixtures import ScriptBackend, PiecesTokenizer, chat, invoke, token_events

LOOPBACK = "127.0.0.1,localhost,[::1]"


def post(host, allowed=None, method="POST", path="/v1/chat/completions"):
    """One request through the real handler with the given Host header (None: no header)."""
    body = json.dumps(chat()).encode()
    headers = [("Content-Length", str(len(body))), ("Content-Type", "application/json")]
    if host is not None:
        headers.insert(0, ("Host", host))
    backend = ScriptBackend(token_events(["hi"]))
    options = None if allowed is None else dict(allowed_hosts=allowed)
    if method == "GET":
        return invoke(backend, PiecesTokenizer(["hi"]), path="/v1/models", method="GET", raw=b"", headers=headers,
                      service_options=options)
    return invoke(backend, PiecesTokenizer(["hi"]), path=path, raw=body, headers=headers, service_options=options)


def status(response):
    return response.statuses[0]


class DefaultOpenTest(unittest.TestCase):
    def test_proxy_host_is_accepted_by_default(self):
        # llama-swap keeps the client's Host header on the upstream request.
        for method in ("POST", "GET"):
            for host in ("myserver:8080", "llama-swap.lan", "10.0.0.5:8080", "[fe80::1]:8080", "127.0.0.1:8790"):
                with self.subTest(method=method, host=host):
                    self.assertEqual(status(post(host, method=method)), 200)

    def test_missing_host_is_accepted_by_default(self):
        self.assertEqual(status(post(None)), 200)
        self.assertEqual(status(post(None, method="GET")), 200)

    def test_utility_endpoint_follows_the_same_rule(self):
        # read_body() checks the Host for count_tokens, tokenize and apply-template too.
        for allowed, expected in ((None, 200), (parse_allowed_hosts("localhost"), 403)):
            with self.subTest(allowed=allowed):
                self.assertEqual(status(post("proxy:80", allowed, path="/tokenize")), expected)


class StrictTest(unittest.TestCase):
    strict = parse_allowed_hosts(LOOPBACK)

    def test_unlisted_host_is_refused_with_text_that_names_the_switches(self):
        for method in ("POST", "GET"):
            with self.subTest(method=method):
                response = post("myserver:8080", self.strict, method=method)
                self.assertEqual(status(response), 403)
                message = json.loads(response.body)["error"]["message"]
                self.assertIn("CRUCIBLE_ALLOWED_HOSTS", message)
                self.assertIn("--allowed-host", message)
                self.assertNotIn("myserver", message)  # client text is not echoed

    def test_listed_hosts_are_accepted_on_any_port(self):
        for host in ("127.0.0.1:8790", "127.0.0.1:1234", "127.0.0.1", "localhost:8790", "[::1]:8790", "[::1]"):
            with self.subTest(host=host):
                self.assertEqual(status(post(host, self.strict)), 200)
                self.assertEqual(status(post(host, self.strict, method="GET")), 200)

    def test_match_ignores_case(self):
        allowed = parse_allowed_hosts("MyServer:8080")
        for host in ("myserver:8080", "MYSERVER:8080", "MyServer:8080"):
            with self.subTest(host=host):
                self.assertEqual(status(post(host, allowed)), 200)

    def test_host_with_a_port_matches_only_that_port(self):
        allowed = parse_allowed_hosts("myserver:8080")
        self.assertEqual(status(post("myserver:8080", allowed)), 200)
        self.assertEqual(status(post("myserver:9090", allowed)), 403)
        self.assertEqual(status(post("myserver", allowed)), 403)

    def test_host_without_a_port_matches_any_port(self):
        allowed = parse_allowed_hosts("myserver")
        for host in ("myserver:8080", "myserver:1", "myserver"):
            with self.subTest(host=host):
                self.assertEqual(status(post(host, allowed)), 200)

    def test_look_alike_hosts_are_refused(self):
        for host in ("evil.com", "localhost.evil.com:8790", "127.0.0.1.evil.com", "xlocalhost", "[::2]:8790",
                     "localhost@evil.com", "127.0.0.1:8790, evil.com", "", " "):
            with self.subTest(host=host):
                self.assertEqual(status(post(host, self.strict)), 403)

    def test_missing_host_is_refused(self):
        self.assertEqual(status(post(None, self.strict)), 403)
        self.assertEqual(status(post(None, self.strict, method="GET")), 403)

    def test_refused_request_never_reaches_the_engine(self):
        backend = ScriptBackend(token_events(["hi"]))
        invoke(backend, PiecesTokenizer(["hi"]), chat(), headers=[("Host", "evil.com")],
               service_options=dict(allowed_hosts=self.strict))
        self.assertEqual(backend.submissions, [])


class ParseTest(unittest.TestCase):
    def test_comma_list_and_blanks(self):
        self.assertEqual(parse_allowed_hosts(" a , B:80 ,,"), (("a", None), ("b", "80")))

    def test_ipv6(self):
        self.assertEqual(parse_allowed_hosts("[::1],[FE80::1]:9"), (("[::1]", None), ("[fe80::1]", "9")))

    def test_empty_means_not_set(self):
        for text in ("", "  ", ",", None):
            with self.subTest(text=text):
                self.assertIsNone(parse_allowed_hosts(text))

    def test_bad_entries_are_errors(self):
        for text in ("a b", "http://a", "a/b", "a:", "a:x", "a:99999", ":80", "a:80:90", "[::1", "::1", "*"):
            with self.subTest(text=text), self.assertRaises(HostError):
                parse_allowed_hosts(text)


class ResolveTest(unittest.TestCase):
    def test_default_is_open(self):
        self.assertIsNone(resolve_allowed_hosts([], {}))

    def test_env_turns_strict_mode_on(self):
        self.assertEqual(resolve_allowed_hosts([], {ALLOWED_HOSTS_ENV: "a,b"}), (("a", None), ("b", None)))

    def test_flag_beats_env(self):
        self.assertEqual(resolve_allowed_hosts(["x:1"], {ALLOWED_HOSTS_ENV: "a,b"}), (("x", "1"),))

    def test_flag_is_repeatable_and_takes_a_comma_list(self):
        self.assertEqual(resolve_allowed_hosts(["a", "b:2,c"], {}), (("a", None), ("b", "2"), ("c", None)))

    def test_bad_env_is_an_error_that_names_the_variable(self):
        with self.assertRaises(HostError) as caught:
            resolve_allowed_hosts([], {ALLOWED_HOSTS_ENV: "a b"})
        self.assertIn(ALLOWED_HOSTS_ENV, str(caught.exception))


class CommandLineTest(unittest.TestCase):
    def run_server(self, *arguments, env=None):
        return subprocess.run([sys.executable, "-m", "server", "/nonexistent-checkpoint", *arguments],
                              capture_output=True, text=True, timeout=60, env=env)

    def test_bad_flag_value_stops_the_start(self):
        result = self.run_server("--allowed-host", "a b")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--allowed-host", result.stderr)

    def test_bad_env_value_stops_the_start(self):
        import os
        result = self.run_server(env={**os.environ, ALLOWED_HOSTS_ENV: "http://x"})
        self.assertEqual(result.returncode, 2)
        self.assertIn(ALLOWED_HOSTS_ENV, result.stderr)

    def test_help_names_the_flag_and_the_risk(self):
        result = self.run_server("--help")
        self.assertIn("--allowed-host", result.stdout)
        self.assertIn("rebinding", result.stdout)


class EngineEnvironmentTest(unittest.TestCase):
    def test_the_host_list_is_not_an_engine_switch_and_is_not_reported_as_one(self):
        from server.backend import engine_environment
        env, removed = engine_environment({ALLOWED_HOSTS_ENV: "a", "CRUCIBLE_FOO": "1", "PATH": "/bin"})
        self.assertEqual(removed, ["CRUCIBLE_FOO"])
        self.assertNotIn(ALLOWED_HOSTS_ENV, env)


if __name__ == "__main__":
    unittest.main()
