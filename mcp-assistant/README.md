# Lica MCP Assistant

A sophisticated assistant for the Lica Model Context Protocol (MCP) server that provides an intelligent, conversational interface for AI-powered multimedia generation and design operations. Built with GPT orchestration and Streamlit web interface, it dynamically executes tools based on user requests and previous results.

## Installation

### Using Docker (Recommended)

1. Clone the repository and navigate to the directory:
```bash
git clone <repository-url>
cd mcp-assistant
```

2. Set up environment variables:
```bash
# Create .env file with your API keys
cat > .env << EOF
ANTHROPIC_API_KEY=your_anthropic_api_key
LICA_MCP_URL=http://localhost:8000/mcp
EOF
```

3. Build and run the Docker container:
```bash
# Build the Docker image
docker build -t mcp-assistant .

# Run the container
docker run -d --name mcp-assistant -p 8501:8501 --env-file .env mcp-assistant
```

The application will start on `http://localhost:8501`.

### Manual Installation (Development)

1. Clone the repository and navigate to the directory:
```bash
git clone <repository-url>
cd mcp-assistant
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create .env file with your API keys
cat > .env << EOF
ANTHROPIC_API_KEY=your_anthropic_api_key
LICA_MCP_URL=https://mcp-server.lica.world/mcp
EOF
```

4. Run the Streamlit application:
```bash
streamlit run app.py
```

## Configuration

### Environment Variables

- `ANTHROPIC_API_KEY`: Anthropic API key for GPT decision engine
- `LICA_MCP_URL`: URL of the Lica MCP server (default: https://mcp-server.lica.world/mcp)

### Application Settings

- **Port**: 8501 (Streamlit default)
- **Host**: localhost (configurable in Streamlit)
- **OAuth Redirect**: http://localhost:8501
- **Session Management**: File-based with automatic cleanup
- **Max Execution Steps**: 10 (configurable in code)

## Architecture

### Core Components

#### 1. LicaMCPAssistant
Main assistant class that orchestrates the entire workflow:
- Discovers available MCP tools
- Manages execution history
- Handles user requests through dynamic tool orchestration

#### 2. MCPConnectionHandler
Handles MCP server connections and basic operations:
- Creates transport connections with proper headers
- Discovers available tools from MCP server
- Executes individual tools with authentication support

#### 3. GPTDecisionEngine
Intelligent decision-making system using Anthropic's Claude:
- Analyzes user requests and execution history
- Decides which tool to call next
- Determines when tasks are complete
- Handles error recovery and information requests

#### 4. ExecutionOrchestrator
Manages the step-by-step execution process:
- Tracks execution history
- Handles different decision actions
- Manages step numbering and status tracking
- Implements maximum step limits

### OAuth Integration

The application includes a complete OAuth 2.0 PKCE flow for Canva integration:

1. **Discovery**: Automatically discovers OAuth endpoints
2. **Client Registration**: Dynamic client registration if needed
3. **PKCE Generation**: Secure code challenge/verifier generation
4. **Token Exchange**: Secure token exchange with refresh support
5. **Session Management**: Persistent token storage and cleanup

## Available Tools

The assistant automatically discovers and orchestrates all tools available from the Lica MCP server.

## Usage Examples

### Basic Chat Interface
```python
# Start the application
streamlit run app.py

# In the web interface, simply type:
"Generate a beautiful sunset image and create a voiceover for it"
```

### Programmatic Usage
```python
from assistant import LicaMCPAssistant

# Initialize assistant
assistant = LicaMCPAssistant()

# Discover available tools
tools = await assistant.discover_tools()

# Handle user request
response = await assistant.handle_user_request(
    "Create a Halloween-themed birthday invitation"
)
```

### Multi-Step Workflows
The assistant automatically handles complex workflows:

1. **User Request**: "Create a presentation about AI and generate a voiceover"
2. **Step 1**: Discover available Canva tools
3. **Step 2**: Generate presentation design
4. **Step 3**: Create voiceover audio
5. **Step 4**: Combine and present results

## Development

### Project Structure
```
mcp-assistant/
├── app.py                 # Main Streamlit application
├── assistant.py           # Lica MCP assistant and orchestration logic
├── system_prompt.txt      # GPT decision engine prompt
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### Development Setup

#### Using Docker (Recommended)

1. **Clone and setup:**
```bash
git clone <repository-url>
cd mcp-assistant
```

2. **Environment configuration:**
```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env
```

3. **Run in development mode:**
```bash
# Build and run with volume mounting for live code changes
docker build -t mcp-assistant .
docker run -d --name mcp-assistant-dev -p 8501:8501 --env-file .env -v $(pwd):/app mcp-assistant

```

#### Manual Setup (Alternative)

1. **Clone and setup:**
```bash
git clone <repository-url>
cd mcp-assistant
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Environment configuration:**
```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env
```

3. **Run in development mode:**
```bash
streamlit run app.py
```

### Adding New Features

#### 1. Custom Tool Integration
```python
# In assistant.py, extend the MCPConnectionHandler
class CustomMCPHandler(MCPConnectionHandler):
    async def execute_custom_tool(self, tool_name: str, parameters: dict):
        # Custom tool execution logic
        pass
```

#### 2. Enhanced UI Components
```python
# In app.py, add custom Streamlit components
def render_custom_ui():
    st.custom_component("custom_widget")
    # Custom UI logic
```

#### 3. Additional OAuth Providers
```python
# Extend OAuth handling in app.py
def _initiate_custom_oauth():
    # Custom OAuth flow
    pass
```

### Customizing the GPT Decision Engine

#### 1. Modify System Prompt
Edit `system_prompt.txt` to change how the AI makes decisions:
- Add new decision rules
- Modify tool selection criteria
- Update response formatting

#### 2. Add Custom Decision Logic
```python
# In assistant.py, extend GPTDecisionEngine
class CustomGPTEngine(GPTDecisionEngine):
    def decide_next_step(self, user_request, execution_history, available_tools):
        # Custom decision logic
        return super().decide_next_step(user_request, execution_history, available_tools)
```

### Error Handling

#### 1. Connection Errors
```python
# Handle MCP server connection issues
try:
    tools = await client.discover_tools()
except ConnectionError:
    st.error("Failed to connect to MCP server")
```

#### 2. Tool Execution Errors
```python
# Handle tool execution failures
status, result = await mcp_handler.execute_tool(tool_name, parameters)
if status == StepStatus.FAILED:
    # Handle error appropriately
    pass
```

#### 3. OAuth Errors
```python
# Handle OAuth flow errors
try:
    tokens = _handle_oauth_callback(code, state)
except ValueError as e:
    st.error(f"OAuth error: {str(e)}")
```

### Testing

#### 1. Unit Tests
```python
# Test individual components
def test_mcp_connection():
    handler = MCPConnectionHandler("http://localhost:8000/mcp")
    # Test connection logic

def test_gpt_decision():
    engine = GPTDecisionEngine()
    # Test decision logic
```

#### 2. Integration Tests
```python
# Test full workflow
async def test_full_workflow():
    assistant = LicaMCPAssistant()
    response = await assistant.handle_user_request("Test request")
    assert "success" in response.lower()
```

#### 3. UI Tests
```python
# Test Streamlit interface
def test_ui_components():
    # Test UI rendering and interactions
    pass
```

### Performance Optimization

#### 1. Caching
```python
# Cache tool discovery results
@lru_cache(maxsize=128)
def get_cached_tools():
    return discover_tools()
```

#### 2. Async Operations
```python
# Use async for I/O operations
async def process_multiple_requests(requests):
    tasks = [handle_request(req) for req in requests]
    return await asyncio.gather(*tasks)
```

#### 3. Memory Management
```python
# Clean up temporary files
def cleanup_temp_files():
    for pattern in ["pkce_*.json", "client_*.json"]:
        for file_path in glob.glob(pattern):
            if is_old_file(file_path):
                os.remove(file_path)
```

### Debugging

#### 1. Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 2. Streamlit Debug Mode
```bash
streamlit run app.py --logger.level=debug
```

#### 3. MCP Server Testing
```bash
# Test MCP server connection
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"method": "tools/list"}'
```

## API Reference

### LicaMCPAssistant

#### `__init__()`
Initialize the Lica MCP assistant with server URL and configuration.

#### `discover_tools() -> List[Dict[str, Any]]`
Discover all available tools from the MCP server.

**Returns:** List of available tools with names, descriptions, and parameters.

#### `execute_tool(tool_name: str, parameters: Dict[str, Any], auth_headers: Optional[Dict[str, str]] = None) -> Tuple[StepStatus, str]`
Execute a single tool with the given parameters.

**Parameters:**
- `tool_name`: Name of the tool to execute
- `parameters`: Tool parameters
- `auth_headers`: Optional authentication headers

**Returns:** Tuple of (status, result)

#### `handle_user_request(user_request: str, auth_headers: Optional[Dict[str, str]] = None) -> str`
Handle a user request through dynamic tool orchestration.

**Parameters:**
- `user_request`: The user's request
- `auth_headers`: Optional authentication headers

**Returns:** Response to the user

### MCPConnectionHandler

#### `discover_tools() -> List[Dict[str, Any]]`
Discover all available tools from the MCP server.

#### `execute_tool(tool_name: str, parameters: Dict[str, Any], auth_headers: Optional[Dict[str, str]] = None) -> Tuple[StepStatus, str]`
Execute a single tool and return the result.

### GPTDecisionEngine

#### `decide_next_step(user_request: str, execution_history: List[ExecutionStep], available_tools: List[Dict[str, Any]]) -> Dict[str, Any]`
Use GPT to decide what to do next based on the user request and execution history.

## Troubleshooting

### Common Issues

1. **MCP Server Connection Failed**
   - Verify `LICA_MCP_URL` is correct
   - Check if MCP server is running
   - Ensure network connectivity

2. **OAuth Authentication Issues**
   - Clear browser cache and cookies
   - Check OAuth redirect URI configuration
   - Verify client credentials

3. **Tool Execution Failures**
   - Check tool parameters are correct
   - Verify required authentication headers
   - Review MCP server logs

4. **GPT Decision Errors**
   - Verify `ANTHROPIC_API_KEY` is set
   - Check API quota and limits
   - Review system prompt configuration

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
streamlit run app.py
```

### Health Check
```bash
# Test MCP server connection
curl http://localhost:8000/health

# Test Streamlit app
curl http://localhost:8501
```

## License

This project is created for educational purposes as part of the LICA @ HackMIT workshop. 

**For Workshop Participants:**
- You are free to use, modify, and distribute this code for educational and learning purposes
- Feel free to experiment, build upon, and create derivative works
- Attribution to the original authors is appreciated but not required
- This code is provided "as is" without any warranties

**For Commercial Use:**
- Please contact the Lica team for commercial licensing terms
- Commercial use requires explicit permission

This workshop material is designed to help you learn about Model Context Protocols (MCPs) and AI-powered multimedia generation. Happy coding! 🚀
