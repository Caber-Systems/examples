# Caber Community Preview: the agentic-tools demo

<!-- profile:begin -->
<p align="center">
  <a href="https://caber.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Caber-Systems/.github/main/images/caber-logo-dark-mode.svg">
      <img src="https://raw.githubusercontent.com/Caber-Systems/.github/main/images/caber-logo-light-mode.svg" alt="Caber" width="180">
    </picture>
  </a>
</p>

<h1 align="center">Active Data Use Control for Agentic AI</h1>
<p align="center"><i>Freshness, provenance, and policy, checked down to the sentence before agents see it.</i></p>

<p align="center">
  <strong><a href="https://github.com/Caber-Systems/examples">Try the free preview</a></strong> ·
  <a href="https://docs.caber.com">Read the docs</a> ·
  <a href="https://caber.com">Caber website</a>
</p>

***
<!-- profile:end -->

Try the Caber Community Preview for free, whether you’re curious or building a proof of concept. Use only non-production infrastructure and data. The Preview limits events, distinct customer objects, and concurrent SDK instances. At a limit, your application can keep running without SDK interception or governance.

**What is here.** Runnable examples of Caber governing what AI agents see. There is one today:

- `demos/agentic-tools/`: a LangGraph agent chooses among four MCP tools and writes its own
  vector store queries over a small corpus of contracts. Caber governs every tool result, so what
  the agent sees depends on who asked, how fresh the content is, and whether it is a copy. You
  run it first in observe mode, where Caber records its decisions, then in enforce mode, where
  Caber redacts.

`./demo.py` at the root of this repository checks, builds, runs and switches the examples;
`runner/` holds its code and `common/` holds the chat app and MCP server the demo uses. The
quick start below takes you from the requirements to a running demo in six commands.

**Licensing.** Caber's code in this repository (`demo.py`, `runner/`, the chat app and MCP
server sources in `common/`, and the demo files in `demos/`) is MIT licensed; see `LICENSE`. Two
things the demo downloads are licensed under the Caber Community Preview Edition License instead
(see License below and `LICENSE-PREVIEW.md`): the `pycaber` SDK, which the build installs from
PyPI into the two demo app images, and the `caber-core` container image. Every other image the
demo pulls (the Caber Console, mock-idp, OpenSearch, PostgreSQL, VersityGW, ElasticMQ and the
`python` base image) and every other package the build installs is licensed by its owner.

**Before you start.** The demo runs on macOS or Linux, on x86_64 or arm64. Windows is not
supported: `./demo.py check` stops there and says so. You need:

- Docker Desktop (macOS or Linux) or Docker Engine (Linux) with Compose 2.20 or newer, and at
  least 16 GB of memory given to Docker
- 20 GB of free disk for the images and volumes
- Python 3.10 or newer; the runner uses the standard library only, so there is nothing to install
- `git`
- an OpenAI API key that can use the chat model `gpt-6-luna` and the embedding model
  `text-embedding-3-small`

The LLM key is your own. The chat app and the MCP server call OpenAI with it, so a run costs you
tokens; "Numbers from the last green run" below lists how many the last CI run used and estimates
the cost at list prices. The demo also runs with Anthropic, NVIDIA, Gemini or Groq; see "Other
LLM providers" after the quick start.

## Quick Start: Six Commands

Run the six steps in order, in one terminal window. Step 2 sets your OpenAI key in that window
and steps 3 to 6 read it from there, so keep the window open until step 6 has finished.

### 1. Clone this repository

```bash
git clone https://github.com/Caber-Systems/examples
cd examples
```

**What you will see.** Git prints `Cloning into 'examples'...` and returns to the prompt.

**Created on your machine:** the directory `examples/` inside the directory you ran the command
from. It holds this repository, and every later step runs from inside it.

### 2. Set your OpenAI key and check this machine

Replace `<your OpenAI API key>` with your key, then run the five lines:

```bash
export SELECTED_LLM_PROVIDER=OPENAI
export OPENAI_API_KEY="<your OpenAI API key>"
export OPENAI_MODEL=gpt-6-luna
export OPENAI_EMBEDDING_MODEL=text-embedding-3-small
./demo.py check
```

**What you will see.** `check` tests the operating system, Docker and the memory it has, Compose,
Python, free disk, the ports 3000, 8050, 8082 and 8083, and your key. To test the key it sends one
short request to `gpt-6-luna` and one to `text-embedding-3-small`, which costs a few tokens. On a
machine that is ready it prints one warning and then `check passed.`:

```
warning: 3 image(s) in preview.lock.json pull anonymously from docker.io, which rate-limits anonymous pulls: opensearchproject/opensearch:2.19.5, postgres:14-alpine, softwaremill/elasticmq-native:1.6.11. If a pull fails, run docker login and try again.
check passed.
```

The warning is advice and needs no action. If `check` prints anything else, each line names one
problem and the last line is `Help: https://github.com/Caber-Systems/examples/discussions`. Fix
what the lines name and run `./demo.py check` again until it prints `check passed.`

**Created on your machine:** Python writes `__pycache__/` folders inside `examples/runner/`. The
four `export` lines change no file; they last as long as this terminal window.

### 3. Build the demo

```bash
./demo.py build
```

**What you will see.** The first line is the license the preview is used under, with the version
and link shown under License below:

```
This preview is governed by the Caber Community Preview Edition License <version>: <link>
```

Then Docker's progress as it pulls the seven images pinned in `preview.lock.json`, then Docker's
output as it builds the two demo app images. Inside that build Docker downloads the
`python:3.12-slim` base image and `pip` installs `pycaber` from PyPI and each app's
`requirements.txt`. `build` returns to the prompt without a closing message when it succeeds;
"Numbers from the last green run" shows how long the last CI build took.

**Created on your machine,** all inside Docker and none in `examples/`:

- the seven pulled images: `caber-core`, `caber-console`, `mock-idp`, VersityGW, OpenSearch,
  PostgreSQL and ElasticMQ, listed with their digests under "Numbers from the last green run"
- the two built images `caber-preview/agentic-chat-app:local` and `caber-preview/mcp-server:local`
- the base image `python:3.12-slim`
- Docker's build cache for the two builds

### 4. Run the demo

```bash
./demo.py run
```

**What you will see.** `run` repeats the machine checks, writes `.env.local`, tests your key
again with the same two short requests, and starts the demo in observe mode. You see Docker's
output as the containers start, then the output of the loader that puts the demo documents into
the stores, then `policies: nothing to activate for this demo.` `run` then waits for the chat
app, the Console and the dashboard to answer, which takes a few minutes; "Numbers from the last
green run" shows how long the last CI run took. Then it opens three tabs in your default browser
and prints:

```
console: http://localhost:3000
dashboard: http://localhost:8050
chat: http://localhost:8083
What to try:
  - Which service categories appear in my contracts and what security controls do they require?
  - Break down the monthly pricing and payment terms listed across my contract documents.
  - Describe the automatic renewal clauses and termination conditions in my agreements.
  - Summarize the scope of services and compliance obligations found in my available contracts.
  - What are the billing structures and total monthly costs for contracts I have access to?
```

Go to the chat tab, http://localhost:8083, and sign in with the email `amy@demo.local` and the
password `cats123`. Ask the first question from the list. The agent chooses among four MCP tools
and writes its own vector store queries, and Caber governs every tool result. In observe mode the
answer goes through unchanged, and the Console tab, http://localhost:3000, records what Caber
would have redacted, and why. Look for three kinds of decision:

- **F2U deny**: an answer that draws on a contract belonging to another demo user. Caber's
  file to user policy denies that user the content.
- **Stale**: an answer that draws on a superseded version of a document when a newer one exists.
- **Duplicate**: the same content found again in a copy stored somewhere else.

![Observe mode: the answer goes through and the Console records the decision](images/preview/observe.png)

The dashboard tab, http://localhost:8050, is the service the Console reads from; you do not need
it for the demo.

**Created on your machine:**

- `examples/.env.local`, readable only by you (file mode 0600): a copy of `.env.template` with
  four freshly generated local secrets, `CSI_HMAC_KEY`, `CSI_SEED_API_KEY`,
  `CSI_OBJECT_STORE_SECRET` and `MOCK_IDP_SCIM_TOKEN`. It does not hold your OpenAI key. Later
  runs reuse it and nothing rewrites it.
- In Docker, under the Compose project `agentic-tools`: twelve running containers, the network
  `agentic-tools_default`, and the volumes `agentic-tools_caber-demo-usage`,
  `agentic-tools_mcp-milvus-lite`, `agentic-tools_versity-s3-data`,
  `agentic-tools_versity-versioning-data`, `agentic-tools_versity-webhook-spool` and
  `agentic-tools_caber-shared`, plus unnamed volumes that some of the images declare. The loader
  runs in a thirteenth container that is removed when it finishes.
- Open ports: the containers listen on ports 3000, 5432, 5556, 7576, 8050, 8082, 8083, 8090,
  9000, 9200 and 9324 on every network interface of this machine, so run the demo only on a
  network you trust.
- Three browser tabs: the Console, the dashboard and the chat app.

### 5. Switch to enforce mode

```bash
./demo.py enforce
```

**What you will see.** `enforce` restarts the chat app and the MCP server with redaction on, waits
until they answer, opens the chat tab again and prints:

```
Mode: enforce (CSI_GLOBAL_REDACT_ENABLED=true).
```

In the chat tab, sign in again with `amy@demo.local` and `cats123` if the sign in form is shown,
and ask the same question as in step 4. Caber now redacts the denied, stale and duplicate content
before the agent sees it, and the Console shows the same decision as enforced:

![Enforce mode: the same question, with the governed content redacted](images/preview/enforce.png)

**Created on your machine:** nothing new. The two restarted containers replace the ones they
were, and the chat tab opens again.

### 6. Stop the demo and remove it from Docker

```bash
./demo.py down
```

**What you will see.** `down` first prints the tokens the demo used since step 4, for example:

```
LLM usage this installation:
  openai: <n> requests, <n> prompt tokens, <n> completion tokens, <n> embedding tokens
```

Then Docker's output as it stops and removes the containers, and last:

```
Help: https://github.com/Caber-Systems/examples/discussions
```

**Removed from your machine:** the containers, the `agentic-tools_default` network, every volume
from step 4, and the nine images from step 3 (seven pulled, two built). To read the token counts,
`down` copies the usage volume into a temporary directory named `caber-demo-usage-` plus a random
suffix in your system's temporary directory, and deletes it before it exits.

**Still on your machine afterwards:** the `examples/` directory with `.env.local` and the
`__pycache__/` folders inside it, the `python:3.12-slim` base image, Docker's build cache, and the
browser tabs. The next section says how to remove each one.

## Everything the demo puts on your machine

| What | Where | Created by | Removed by |
| --- | --- | --- | --- |
| This repository | `examples/` | step 1, `git clone` | deleting `examples/` |
| `__pycache__/` folders | inside `examples/runner/` | the first `./demo.py` command | deleting `examples/` |
| `.env.local`, mode 0600, four generated local secrets | `examples/.env.local` | the first `./demo.py run` | `./demo.py down --purge`, or deleting `examples/` |
| Seven pinned images | Docker | `./demo.py build` | `./demo.py down` |
| `caber-preview/agentic-chat-app:local`, `caber-preview/mcp-server:local` | Docker | `./demo.py build` | `./demo.py down` |
| `python:3.12-slim` base image | Docker | `./demo.py build` | `docker image rm python:3.12-slim`, unless you had it before |
| Build cache of the two app images | Docker | `./demo.py build` | `docker builder prune` |
| Twelve containers and the network `agentic-tools_default` | Docker | `./demo.py run` | `./demo.py down` |
| Six named volumes `agentic-tools_*` and the unnamed volumes of the images | Docker | `./demo.py run` | `./demo.py down` |
| Temporary directory `caber-demo-usage-*` | your system's temporary directory | `./demo.py down` | `./demo.py down`, before it exits |
| Three browser tabs | your default browser | `./demo.py run`; `enforce` and `observe` reopen the chat tab | closing them |

Nothing is written outside `examples/`, Docker, your temporary directory and your browser.

## demo.py command summary

Run every command from the root of this repository. `./demo.py --help` lists them and
`./demo.py <command> --help` describes one.

| Command | What it does |
| --- | --- |
| `./demo.py check` | Checks this machine and your LLM key; changes nothing. |
| `./demo.py build` | Pulls the images pinned in `preview.lock.json` and builds the two demo app images. |
| `./demo.py run` | Writes `.env.local` the first time, starts the demo in observe mode, loads the documents and opens the browser tabs. |
| `./demo.py enforce` | Switches the running demo to enforce mode: Caber redacts. |
| `./demo.py observe` | Switches the running demo back to observe mode: decisions are recorded and nothing is redacted. |
| `./demo.py down` | Prints the LLM tokens used, then removes the containers, network, volumes and images. |
| `./demo.py down --keep-images` | The same, but keeps the nine images, so the next `build` has nothing to pull. |
| `./demo.py down --purge` | The same as `down`, and also deletes `.env.local`; the next `run` generates new secrets. |

`build` also accepts `--pycaber`, `--tag` and `--i-know-this-is-untested`. They replace what
`preview.lock.json` certified with versions nobody tested together, so the quick start leaves
them off.

### Where the runner reads settings

Every command reads variables from the shell and from `examples/.env.local`. An existing `.env`
beside `.env.template` is read too. The shell wins over `.env.local`, which wins over `.env`.
`.env.template` is the list of every setting with its default. It is not read directly: `run`
copies it into `.env.local`, so edit `.env.local`, never `.env.template`. To keep your key across
terminal windows, add the line `OPENAI_API_KEY=<your key>` (and `SELECTED_LLM_PROVIDER=OPENAI`) to
`.env.local` after the first `run`.

### Other LLM providers

Set `SELECTED_LLM_PROVIDER` to `ANTHROPIC`, `NVIDIA`, `GEMINI`, `OPENAI` or `GROQ`, and set that
provider's `<PROVIDER>_API_KEY` and `<PROVIDER>_MODEL` (for example `GEMINI_API_KEY` and
`GEMINI_MODEL`). NVIDIA, Gemini and OpenAI also serve the embeddings and need
`<PROVIDER>_EMBEDDING_MODEL`. Anthropic and Groq have no embeddings, so with them also set
`SELECTED_EMBEDDING_PROVIDER` to `NVIDIA`, `GEMINI` or `OPENAI`, plus that provider's key and
embedding model. `.env.template` lists every variable with the default model of each provider.

### More documentation

The Caber documentation for this demo, the SDK and the Community Preview is at
https://docs.caber.com/demos-and-benchmarks/.

## Numbers from the last green run

<!-- lock:begin -->
Measured on 2026-11-18

Recorded in `preview.lock.json` by the last green CI run.

pycaber 0.2.5683 from https://pypi.org/simple

| Image | Digest |
| --- | --- |
| `ghcr.io/caber-systems/caber-core:0.2.5683` | `sha256:b359ae31426f` |
| `ghcr.io/versity/versitygw:v1.7.0` | `sha256:c4cbd9d9cb8d` |
| `opensearchproject/opensearch:2.19.5` | `sha256:4ee82ecb35d8` |
| `softwaremill/elasticmq-native:1.6.11` | `sha256:fe3af4a8dd59` |
| `ghcr.io/caber-systems/caber-console` | `sha256:32d73e1ae3e7` |
| `ghcr.io/caber-systems/mock-idp:preview` | `sha256:000000000000` |

Durations of that run, in seconds:

- build: 2 s
- run_to_healthy: 45 s
- ci_total: 9 s

LLM tokens that run used, per provider:

| Provider | Chat model | Embedding model | Chat input | Chat output | Embedding tokens | Requests |
| --- | --- | --- | --- | --- | --- | --- |
| anthropic | claude-sonnet-5-5 | nvidia/nemotron-3-embed-1b | 142328 | 10156 | 106752 | 673 |
| gemini | gemini-3.8-flash | nvidia/nemotron-3-embed-1b | 118739 | 4837 | 106670 | 667 |
| groq | openai/gpt-oss-120b | nvidia/nemotron-3-embed-1b | 97785 | 10406 | 106729 | 671 |
| nvidia | meta/muse-glimmer-30b | nvidia/nemotron-3-embed-1b | 108532 | 15051 | 106788 | 676 |
| openai | gpt-6-luna | nvidia/nemotron-3-embed-1b | 81956 | 3660 | 106840 | 672 |

### List prices as published on 2026-11-05

In USD per million tokens. This table is a fixed copy kept in the exporter; check the provider's own page before you rely on it.

| Provider | Model | Input | Output |
| --- | --- | --- | --- |
| anthropic | claude-sonnet-5-5 | price not in table | price not in table |
| nvidia | nvidia/nemotron-3-embed-1b | price not in table | n/a |
| gemini | gemini-3.8-flash | price not in table | price not in table |
| groq | openai/gpt-oss-120b | 0.15 | 0.75 |
| nvidia | meta/muse-glimmer-30b | price not in table | price not in table |
| openai | gpt-6-luna | price not in table | price not in table |

Estimated cost of one run at those prices:

- anthropic: price not in table
- gemini: price not in table
- groq: price not in table
- nvidia: price not in table
- openai: price not in table

Caber Community Preview Edition License v1.2: https://raw.githubusercontent.com/Caber-Systems/preview-license/v1.2/LICENSE-PREVIEW.md
<!-- lock:end -->

## Help

Ask questions and share what you built in GitHub Discussions:
https://github.com/Caber-Systems/examples/discussions (the Q&A and Show-and-tell categories).
`./demo.py down` prints the same link. For licensing questions, write to support@caber.com.

## License

<!-- notice:begin -->
Caber's code in this repository is MIT licensed (LICENSE), the pycaber SDK from PyPI and the caber-core container image are licensed under the [Caber Community Preview Edition License v1.2](https://raw.githubusercontent.com/Caber-Systems/preview-license/v1.2/LICENSE-PREVIEW.md) (shipped here as LICENSE-PREVIEW.md), and every other image and package the demo pulls or installs is licensed by its owner.
<!-- notice:end -->
