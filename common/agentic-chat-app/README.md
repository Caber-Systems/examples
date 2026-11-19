# agentic-chat-app

The Streamlit chat app (http://localhost:8083). A LangGraph agent chooses among the MCP tools and
writes its own vector-store queries; the pycaber SDK governs every tool result before the agent
sees it. The app calls your LLM provider with the key you set for `./demo.py`.

`./demo.py build` builds this directory into the chat app image; `Dockerfile` is the recipe.
