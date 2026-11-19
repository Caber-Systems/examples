"""Shared Milvus semantic-search query path.

Ticket 06 decided that the MCP server should **embed** semantic search via a
shared query module rather than proxying to the ragchat service: a proxy adds a
network hop and a second service whose health can fail the demo, for no benefit,
since provenance is captured agent-side at the framework tool wrapper and so does
not care where the query executes.

This module exists rather than importing the (now archived)
``CaberApp/milvus-ragchat-app``'s
``encoder.py`` / ``milvus_utils.py`` directly, because those are **Streamlit-
coupled** -- their functions are wrapped in ``@st.cache_resource``, so importing
them drags Streamlit into the MCP server and ties a cache lifecycle to a UI
session that does not exist there.

So: this is the canonical, framework-free implementation of "embed a query, search
a collection". It is deliberately small -- embedding and search only. Collection
creation and corpus insertion stay where they already work.

Known follow-up, recorded rather than done: the archived ``milvus-ragchat-app`` still has its
own copies of ``emb_text`` and ``get_search_results``. They should be thinned to
wrappers over this module, but that means editing a working demo's code for no
functional gain in this effort, so it is left alone. If you are touching ragchat
anyway, do it then.
"""

from .query import (  # noqa: F401
    MilvusQueryError,
    SemanticHit,
    embed_query,
    get_client,
    search,
    set_usage_recorder,
)
