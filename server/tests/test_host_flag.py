"""--host ADDRESS: the bind address (default 127.0.0.1), an IPv4 or IPv6 literal, with a warning off loopback."""
import ipaddress
import socket
import unittest
from http.server import BaseHTTPRequestHandler
from unittest.mock import patch

import server.__main__ as startup
from server.tests.test_server_defaults import parse


def listening(args):
    """The start lines, and the mock of bind_server: main() calls it with (address, port, handler)."""
    with patch.object(startup, "bind_server") as bind:
        bind.return_value.serve_forever.side_effect = KeyboardInterrupt
        options, err = parse(args)
    return options, err, bind


def ipv6_available():
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::1", 0))
        return True
    except OSError:
        return False


class ParseTest(unittest.TestCase):
    def test_default_binds_127_0_0_1_with_no_warning(self):
        _, err, bind = listening([])
        self.assertEqual(bind.call_args.args[:2], (ipaddress.ip_address("127.0.0.1"), 8790))
        self.assertNotIn("no authentication", err)
        self.assertIn("ready at http://127.0.0.1:8790/v1", err)

    def test_ipv4_literals(self):
        for host in ("127.0.0.1", "127.0.0.2", "0.0.0.0", "192.0.2.7"):
            with self.subTest(host=host):
                _, err, bind = listening(["--host", host, "--port", "9001"])
                self.assertEqual(bind.call_args.args[:2], (ipaddress.ip_address(host), 9001))
                self.assertIn(f"ready at http://{host}:9001/v1", err)

    def test_ipv6_literals(self):
        for host in ("::1", "::", "2001:db8::7"):
            with self.subTest(host=host):
                _, err, bind = listening(["--host", host, "--port", "9001"])
                self.assertEqual(bind.call_args.args[:2], (ipaddress.ip_address(host), 9001))
                self.assertIn(f"ready at http://[{host}]:9001/v1", err)

    def test_bad_values_exit_2_before_loading(self):
        for host in ("localhost", "example.com", "", "256.0.0.1", "1.2.3", "::g", "[::1]", "127.0.0.1:80",
                     "fe80::1%eth0", " 127.0.0.1"):
            with self.subTest(host=host):
                code, _ = parse(["--host", host])
                self.assertEqual(code, 2)


class WarningTest(unittest.TestCase):
    def warnings(self, host):
        _, err, _ = listening(["--host", host])
        return [line for line in err.splitlines() if "no authentication" in line]

    def test_loopback_has_no_warning(self):
        for host in ("127.0.0.1", "127.9.9.9", "::1"):
            with self.subTest(host=host):
                self.assertEqual(self.warnings(host), [])

    def test_other_addresses_have_one_warning(self):
        for host in ("0.0.0.0", "::", "192.0.2.7", "2001:db8::7"):
            with self.subTest(host=host):
                (line,) = self.warnings(host)
                self.assertIn(host, line)
                self.assertIn("--allowed-host", line)
                self.assertIn("firewall", line)


class BindTest(unittest.TestCase):
    def test_the_class_follows_the_address_family(self):
        with patch.object(startup, "ThreadingHTTPServer") as v4, patch.object(startup, "Ipv6Server") as v6:
            startup.bind_server("192.0.2.7", 0, BaseHTTPRequestHandler)
            v6.assert_not_called()
            startup.bind_server("2001:db8::7", 0, BaseHTTPRequestHandler)
            v4.assert_called_once()
            v6.assert_called_once()
        self.assertEqual(startup.Ipv6Server.address_family, socket.AF_INET6)

    def serve(self, host):
        server = startup.bind_server(host, 0, BaseHTTPRequestHandler)
        self.addCleanup(server.server_close)
        return server

    def test_binds_127_0_0_1_on_port_0(self):
        server = self.serve("127.0.0.1")
        self.assertEqual(server.server_address[0], "127.0.0.1")
        self.assertEqual(server.socket.family, socket.AF_INET)
        self.assertGreater(server.server_address[1], 0)

    @unittest.skipUnless(ipv6_available(), "this host has no IPv6 loopback (::1 does not bind)")
    def test_binds_ipv6_loopback_on_port_0(self):
        server = self.serve("::1")
        self.assertEqual(server.server_address[0], "::1")
        self.assertEqual(server.socket.family, socket.AF_INET6)
        self.assertGreater(server.server_address[1], 0)


if __name__ == "__main__":
    unittest.main()
