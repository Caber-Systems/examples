# common

The two demo applications, built on your machine by `./demo.py build`. Each directory is its own
docker build context, and both install the pycaber SDK at the version `preview.lock.json` names.

- `agentic-chat-app/`: the Streamlit chat app and its LangGraph agent
- `mcp-server/`: the MCP tool server the agent calls
