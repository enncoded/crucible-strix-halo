"""Device protocol regression. Run under the shared GPU lock and a timeout."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / 'tools'))  # tools/work_paths.py
import work_paths  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--engine", default="build/engine/crucible")
    parser.add_argument("--baseline", type=Path, default=work_paths.worktree("integrate") / "results/bounds-final/spec0.txt")
    args = parser.parse_args()
    fib = work_paths.fibonacci_ids()
    ids = [int(value) for value in fib.read_text().strip().split(",")]
    expected = [int(value) for line in args.baseline.read_text().splitlines() if line.startswith("ids:")
                for value in line[4:].split()][:8]
    output = Path("results/serve-protocol")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "engine.log").open("w") as log:
        process = subprocess.Popen([args.engine, "serve", args.checkpoint], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        transcript = []
        def receive():
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("The resident engine stopped unexpectedly.")
            transcript.append(json.loads(line))
            return transcript[-1]
        def request(payload):
            process.stdin.write(payload if isinstance(payload, str) else json.dumps(payload))
            process.stdin.write("\n")
            process.stdin.flush()
            tokens = []
            while True:
                event = receive()
                if event["event"] != "token":
                    return tokens, event
                tokens.append(event["id"])
        try:
            assert receive()["event"] == "ready"
            tokens, event = request('{"ids":[')
            assert not tokens and event["event"] == "error"
            normal = dict(ids=ids, max_tokens=8, spec=0, stop_ids=[])
            tokens, event = request(normal)
            assert tokens == expected and event["event"] == "done"
            tokens, event = request(normal | dict(spec=2))
            assert tokens == expected and event["event"] == "done"
            tokens, event = request(normal | dict(spec=2, stop_ids=[expected[0]]))
            assert tokens == expected[:1] and event["finish_reason"] == "stop"
            tokens, event = request(dict(ids=[1], max_tokens=1, spec=0, stop_ids=[]))
            assert len(tokens) == 1 and event["event"] == "done"
            tokens, event = request(normal | dict(spec=2))
            assert tokens == expected and event["event"] == "done"
            process.stdin.close()
            assert process.wait(timeout=30) == 0
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
        (output / "events.json").write_text(json.dumps(transcript, indent=2))
    print("PASS resident JSON errors recover; serial/spec2 IDs, stop rollback, and sequence reset match gen")
    print("PASS resident engine stops after stdin EOF")

if __name__ == "__main__":
    main()
