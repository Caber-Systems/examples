# config

Platform configuration mounted into the containers. config-base.json is the
platform default; config.json, config-mcp.json and config-chat.json are the demo
overlays for csi-init, the MCP server and the chat app. Credentials are
${...} references to the values in .env.local.
