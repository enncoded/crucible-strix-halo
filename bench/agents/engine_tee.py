"""Pass engine stdin/stdout unchanged. Save ready and done events for memory accounting."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def main():
    engine = Path(os.environ['AGENTS_ENGINE'])
    events = Path(os.environ['AGENTS_EVENTS'])
    process = subprocess.Popen([str(engine), *sys.argv[1:]], stdout=subprocess.PIPE)
    def stop(_signal, _frame):
        if process.poll() is None:
            process.terminate()
    signal.signal(signal.SIGTERM, stop)
    try:
        with events.open('xb') as output:
            for line in process.stdout:
                value = json.loads(line)
                if value.get('event') in ('ready', 'done', 'error'):
                    output.write(line)
                    output.flush()
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
        return process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)


if __name__ == '__main__': raise SystemExit(main())
