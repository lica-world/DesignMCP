# MCP Workshop

## Run CanvaMCP on Cursor:

1. Install MCP remote: `npm install -g mcp-remote`
2. Create a Cursor project
3. Create the folder `.cursor`
4. Configure a new MCP server at `.cursor/mcp.json`:
```
{
  "mcpServers": {
    "canva": {
      "command": "npx",
      "args": [
        "-y",
        "mcp-remote@latest",
        "https://mcp.canva.com/mcp"
      ]
    }
  }
}
```

Chat and create!

Tip: start by asking `/list_tools` to see all available tools.

### Live Demo
1. `/list_tools`
2. Create an invitation for a birthday party for a boy who loves dinosaurs.
3. Add a comment to the first design saying that I like it the most.
4.   Note: You shold refer the actual canva url rather than the generation url from the chat.
5. Show me all comments on the design ...

## FastMCP

See example [code](https://github.com/lica-world/DesignMCP/blob/main/mcp-server/server.py).

## FigmaMCP

1. Install Figma Desktop
2. Follow: https://help.figma.com/hc/en-us/articles/32132100833559-Guide-to-the-Dev-Mode-MCP-Server#h_01K20CAZQFATWZ8DH8DFVXDDV8


