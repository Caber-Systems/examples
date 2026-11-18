# agentic-tools

A LangGraph agent chooses among four MCP tools and writes its own Milvus queries over a small
corpus of contracts. Caber governs every tool result at the LangChain tool boundary, so what the
agent sees depends on who asked, how fresh the content is, and whether it is a copy.

Start, stop and switch the demo from the repository root with `./demo.py`; the root README.md has
the quickstart.

## Services

Every service is defined in `docker-compose.yaml` in this directory.

<!-- services:begin -->
- `agentic-chat-app`: The Streamlit chat app (http://localhost:8083): a LangGraph agent that picks among the MCP tools, governed by the pycaber SDK.
- `caber-console`: The Caber Console (http://localhost:3000): policies, audit trail and the governance view.
- `caber-core`: The Caber platform core: ingest, fingerprinting, the intersector and the SDK endpoint.
- `csi-dashboard`: The Caber dashboard API (http://localhost:8050) the Console reads; runs from the caber-core image.
- `csi-dep-objstor`: S3-compatible object store (VersityGW) holding the demo buckets.
- `csi-dep-queue`: SQS-compatible queue (ElasticMQ) carrying object-store events.
- `csi-dep-search`: OpenSearch for the platform's event and audit indices.
- `csi-init`: One-shot platform setup and the config server on :7576; runs from the caber-core image.
- `mcp-server-back-door`: The MCP tool server (http://localhost:8082) the agent calls, with its embedded Milvus Lite vector store.
- `mock-idp`: Mock identity provider holding the demo users (directory in ./identity/directory.json).
- `object-store-tools`: Relays object-store webhooks to the platform; runs from the caber-core image.
- `seed`: One-shot loader: puts the demo corpus into the object store and the vector store (profile seed).
<!-- services:end -->

## What to try

Sign in to the chat app (http://localhost:8083) as one of the demo users and ask, for example:

<!-- what-to-try:begin -->
- Which service categories appear in my contracts and what security controls do they require?
- Break down the monthly pricing and payment terms listed across my contract documents.
- Describe the automatic renewal clauses and termination conditions in my agreements.
- Summarize the scope of services and compliance obligations found in my available contracts.
- What are the billing structures and total monthly costs for contracts I have access to?
<!-- what-to-try:end -->

Ask the same question as two different users, and again after `./demo.py enforce`, then compare
the decisions in the Caber Console (http://localhost:3000).

## What is in this directory

- `demo.json`: what `./demo.py` reads to run this demo
- `docker-compose.yaml`: the single compose file for the whole stack, every pulled image pinned by digest
- `config/`: the platform configuration mounted into the containers
- `identity/`: the demo users the mock identity provider serves
- `corpus/`: the demo documents
- `seed/`: the one-shot loader that puts the corpus into the object store and the vector store
- `governance/`: empty until a policy is activated
- `postgres-initdb/`: SQL PostgreSQL runs once on its first start
