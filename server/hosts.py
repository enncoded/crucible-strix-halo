"""The Host header rule. The server binds 127.0.0.1 only, but a reverse proxy (llama-swap, nginx, LiteLLM) forwards the
client's own Host header, so the default accepts any Host, as llama.cpp, vLLM and Ollama do. The strict list below is the
defence against DNS rebinding for a machine whose browser can reach the server."""
ALLOWED_HOSTS_ENV = "CRUCIBLE_ALLOWED_HOSTS"
MAX_PORT = 65535

REFUSED = (f"The Host header is not allowed. Add its name to {ALLOWED_HOSTS_ENV} (comma-separated) or start the "
           "server with --allowed-host.")


class HostError(ValueError):
    pass


def split_host(text):
    """(lowercase host, port text or None) of a Host header or a list entry. Raises HostError when it is not one."""
    text = text.strip().lower()
    if text.startswith("["):  # an IPv6 literal keeps its brackets: [::1] or [::1]:8790
        end = text.find("]")
        host, rest = text[:end + 1], text[end + 1:]
        if end < 0 or len(host) < 3 or (rest and not rest.startswith(":")):
            raise HostError(f"{text!r} is not a host name")
        port = rest[1:] if rest else None
        if port == "":
            raise HostError(f"{text!r} has an empty port")
    else:
        host, colon, port = text.partition(":")
        port = port if colon else None
        if ":" in (port or ""):
            raise HostError(f"{text!r} is not a host name")
    if not host or any(character.isspace() or character in "/@,*\\?#" for character in host):
        raise HostError(f"{text!r} is not a host name")
    if port is not None and not (port.isascii() and port.isdecimal() and 0 < int(port) <= MAX_PORT):
        raise HostError(f"{text!r} has a port outside 1 to {MAX_PORT}")
    return host, port


def parse_allowed_hosts(text):
    """The strict list of a comma-separated text as ((host, port or None), ...), or None when the text names no host."""
    if text is None:
        return None
    entries = tuple(split_host(entry) for entry in text.split(",") if entry.strip())
    return entries or None


def resolve_allowed_hosts(flags, environ):
    """--allowed-host values (each may be a comma list) win over the environment variable. None: any Host."""
    if flags:
        return parse_allowed_hosts(",".join(flags))
    try:
        return parse_allowed_hosts(environ.get(ALLOWED_HOSTS_ENV))
    except HostError as error:
        raise HostError(f"{ALLOWED_HOSTS_ENV}: {error}") from None


def host_allowed(header, allowed):
    """True when no list is set, or when the Host header matches an entry (a host without a port matches any port)."""
    if allowed is None:
        return True
    if header is None:
        return False
    try:
        host, port = split_host(header)
    except HostError:
        return False
    return any(host == name and (limit is None or limit == port) for name, limit in allowed)
