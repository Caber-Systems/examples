# runner

The Python code behind `./demo.py`. It uses the standard library only, so Python 3.10 or newer is
all it needs.

- `locked/`: the preview runner. It checks `preview.lock.json`, pulls the pinned images, builds the
  two app images, runs the stack with docker compose, and switches between enforce and observe.
- `core/`: shared helpers, such as the policy plan the demo applies.
