# mcp-server

The MCP tool server (http://localhost:8082) the agent calls. It answers from an embedded Milvus
Lite vector store over the demo corpus, and the pycaber SDK governs every tool result.

`./demo.py build` builds this directory into the MCP server image; `Dockerfile` is the recipe and
`shared/milvus_query/` holds the vector-store query helpers it copies in.
