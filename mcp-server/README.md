# Lica MCP Server

A comprehensive Model Context Protocol (MCP) server providing AI-powered tools for multimedia generation, content creation, and design operations. Built with FastMCP, it integrates multiple AI services to offer a complete creative toolkit.

## Features

- **MCP Protocol Support**: Full MCP server implementation with FastMCP
- **AI Text-to-Speech**: ElevenLabs integration for high-quality voice synthesis
- **AI Text-to-Video**: Synthesia integration for automated video generation
- **AI Image Generation**: Kling AI for text-to-image and image-to-video generation
- **AI Background Removal**: Automated background removal from images
- **AI Audio/Video Dubbing**: Multi-language dubbing using ElevenLabs
- **AI Lip Sync**: Video lip synchronization with audio using Kling AI
- **AI Subtitle Generation**: Auto-transcription and subtitle addition using Shotstack + ElevenLabs
- **Canva Integration**: Direct integration with Canva MCP for design operations
- **Multi-Server Registry**: Manage multiple MCP servers from a single endpoint
- **File Serving**: Built-in media file serving (audio, video, images) with public URLs
- **Health Monitoring**: Health check endpoints for service monitoring

## Installation

### Using Docker (Recommended)

1. Clone the repository and navigate to the directory:
```bash
git clone <repository-url>
cd mcp-server
```

2. Set up environment variables:
```bash
# Create .env file with your API keys
cat > .env << EOF
ELEVENLABS_API_KEY=your_elevenlabs_api_key
SYNTHESIA_API_KEY=your_synthesia_api_key
KLING_ACCESS_KEY=your_kling_access_key
KLING_SECRET_KEY=your_kling_secret_key
BACKGROUND_ERASE_API_KEY=your_background_erase_api_key
SHOTSTACK_API_KEY=your_shotstack_api_key
EOF
```

3. Build and run the Docker container:
```bash
# Build the Docker image
docker build -t mcp-server .

# Run the container
docker run -d --name mcp-server -p 8000:8000 --env-file .env mcp-server
```

The server will start on `http://localhost:8000` with MCP endpoint at `/mcp`.

### Manual Installation (Development)

1. Clone the repository and navigate to the directory:
```bash
git clone <repository-url>
cd mcp-server
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create .env file with your API keys
cat > .env << EOF
ELEVENLABS_API_KEY=your_elevenlabs_api_key
SYNTHESIA_API_KEY=your_synthesia_api_key
KLING_ACCESS_KEY=your_kling_access_key
KLING_SECRET_KEY=your_kling_secret_key
BACKGROUND_ERASE_API_KEY=your_background_erase_api_key
SHOTSTACK_API_KEY=your_shotstack_api_key
EOF
```

4. Run the server:
```bash
python server.py
```

## Configuration

### Environment Variables

- `ELEVENLABS_API_KEY`: ElevenLabs API key for text-to-speech and dubbing functionality
- `SYNTHESIA_API_KEY`: Synthesia API key for text-to-video generation
- `KLING_ACCESS_KEY`: Kling AI access key for image and video generation
- `KLING_SECRET_KEY`: Kling AI secret key for JWT authentication
- `BACKGROUND_ERASE_API_KEY`: Background Erase API key for image processing
- `SHOTSTACK_API_KEY`: Shotstack API key for video subtitle processing

### Server Settings

- **Port**: 8000 (configurable in `server.py`)
- **Host**: 0.0.0.0 (all interfaces)
- **Transport**: Streamable HTTP for MCP protocol
- **Media Files**: Stored in temporary directory with public serving
- **File Endpoints**: `/audio/{filename}`, `/image/{filename}`, `/video/{filename}`

## Available Tools

### Core MCP Tools

#### `get_external_server_tools(server_name)`
List all available tools from an external MCP server.

**Parameters:**
- `server_name` (str): Name of the external MCP server (e.g., "canva")

**Returns:** Formatted list of available tools for the specified server.

#### `execute_external_tool(server_name, tool_name, parameters)`
Execute a tool from an external MCP server.

**Parameters:**
- `server_name` (str): Name of the external MCP server (e.g., "canva")
- `tool_name` (str): Name of the tool to execute
- `parameters` (dict): Tool parameters

**Returns:** Result of the tool execution.

### AI Content Generation Tools

#### `text_to_speech(text, voice_id, model_id, ...)`
Convert text to high-quality speech using ElevenLabs AI.

**Parameters:**
- `text` (str): Text to convert to speech
- `voice_id` (str): Voice ID (default: "JBFqnCBsd6RMkjVDRZzb" - George)
- `model_id` (str): AI model (default: "eleven_multilingual_v2")
- `output_format` (str): Audio format (default: "mp3_22050_32")
- `stability` (float): Voice stability 0.0-1.0 (default: 0.5)
- `similarity_boost` (float): Voice similarity 0.0-1.0 (default: 0.8)
- `style` (float): Style exaggeration 0.0-1.0 (default: 0.0)
- `use_speaker_boost` (bool): Enable speaker boost (default: true)
- `speed` (float): Speech speed 0.25-4.0 (default: 1.0)

**Available Voices:**
- `JBFqnCBsd6RMkjVDRZzb` - George (Warm male voice)
- `21m00Tcm4TlvDq8ikWAM` - Rachel (Natural female voice)
- `AZnzlk1XvdvUeBnXmlld` - Domi (Confident female voice)
- `EXAVITQu4vr4xnSDxMaL` - Bella (Expressive female voice)
- `ErXwobaYiN019PkySvjV` - Antoni (Smooth male voice)
- `MF3mGyEYCl7XYWbV9V6O` - Elli (Energetic female voice)
- `TxGEqnHWrfWFTfGW9XjX` - Josh (Deep male voice)
- `VR6AewLTigWG4xSOukaG` - Arnold (Authoritative male voice)
- `pNInz6obpgDQGcFmaJgB` - Adam (Professional male voice)
- `yoZ06aMxZJJ28mfd3POQ` - Sam (Casual male voice)

**Returns:** Public URL to the generated audio file with conversion details.

#### `audio_video_dubbing(file_url, target_language, source_language, voice_id, watermark)`
Dub audio or video content to different languages using ElevenLabs AI.

**Parameters:**
- `file_url` (str): URL or path to the audio/video file to dub
- `target_language` (str): Target language code (default: "es" - Spanish)
- `source_language` (str): Source language (default: "auto" for auto-detection)
- `voice_id` (str, optional): Specific voice ID to use (uses voice cloning if None)
- `watermark` (bool): Whether to include ElevenLabs watermark (default: False)

**Supported Languages:**
- `en` - English, `es` - Spanish, `fr` - French, `de` - German, `it` - Italian
- `pt` - Portuguese, `pl` - Polish, `tr` - Turkish, `ru` - Russian, `nl` - Dutch
- `cs` - Czech, `ar` - Arabic, `zh` - Chinese, `ja` - Japanese, `hu` - Hungarian, `ko` - Korean

**Returns:** Public URL to the dubbed file with processing details.

#### `remove_background(image_url)`
Remove background from an image using AI background removal.

**Parameters:**
- `image_url` (str): URL or path to the image file

**Returns:** Public URL to the processed image with transparency preserved.

#### `text_to_image(prompt, aspect_ratio, image_count, model)`
Generate images from text descriptions using Kling AI.

**Parameters:**
- `prompt` (str): Text description of the image to generate
- `aspect_ratio` (str): Aspect ratio (default: "1:1")
  - `1:1` - Square (1024x1024)
  - `16:9` - Landscape (1344x768)
  - `9:16` - Portrait (768x1344)
  - `4:3` - Standard (1152x896)
  - `3:4` - Vertical (896x1152)
- `image_count` (int): Number of images to generate (default: 1)
- `model` (str): AI model to use (default: "kling-v1")

**Returns:** Public URL to the generated image with generation details.

#### `image_to_video(image_url, prompt, aspect_ratio, duration, model, cfg)`
Convert an image to video using Kling AI image-to-video generation.

**Parameters:**
- `image_url` (str): URL or path to the input image
- `prompt` (str): Optional text prompt to guide video generation
- `aspect_ratio` (str): Aspect ratio (default: "9:16")
  - `16:9` - Landscape (1344x768)
  - `9:16` - Portrait (768x1344)
  - `1:1` - Square (1024x1024)
- `duration` (str): Video duration in seconds (default: "5")
- `model` (str): AI model to use (default: "kling-v1")
- `cfg` (float): Classifier-free guidance scale 0.0-1.0 (default: 0.5)

**Returns:** Public URL to the generated video with generation details.

#### `lip_sync(video_url, audio_url, model)`
Synchronize lip movements in a video with audio using Kling AI lip sync.

**Parameters:**
- `video_url` (str): URL or path to the source video file
- `audio_url` (str): URL or path to the audio file to sync with
- `model` (str): AI model to use (default: "kling-v1")

**Returns:** Public URL to the lip-synced video with processing details.

#### `add_subtitles(video_url, subtitles_content, subtitle_type, output_format, resolution)`
Add subtitles to a video using Shotstack API with auto-transcription support.

**Parameters:**
- `video_url` (str): URL or path to the source video file
- `subtitles_content` (str): Subtitle content (SRT format, plain text, or empty for auto-generation)
- `subtitle_type` (str): Type of subtitle content (default: "auto")
  - `"auto"` - Auto-generate subtitles from video audio using ElevenLabs transcription
  - `"srt"` - Use provided SRT content
  - `"text"` - Use provided plain text
- `output_format` (str): Output video format (default: "mp4")
- `resolution` (str): Output resolution (default: "hd")

**Returns:** Public URL to the subtitled video with processing details.

#### `text_to_video(text, avatar, background, aspect_ratio, title)`
Convert text to professional video using Synthesia AI.

**Parameters:**
- `text` (str): Text for the avatar to speak
- `avatar` (str): Avatar selection (default: "anna_costume1_cameraA")
- `background` (str): Background color/template (default: "#00A2FF")
- `aspect_ratio` (str): Video aspect ratio (default: "9:16")
- `title` (str): Video title (default: "test video")

**Available Avatars:**
- `anna_costume1_cameraA` - Professional female presenter
- `jake_costume1_cameraA` - Professional male presenter
- `maya_costume1_cameraA` - Casual female presenter
- `david_costume1_cameraA` - Business male presenter
- `sarah_costume1_cameraA` - Friendly female presenter

**Background Options:**
- Hex colors: `#00A2FF`, `#FF6B6B`, `#4ECDC4`
- Solid colors: `white`, `black`, `blue`, `green`, `red`
- Special: `green_screen` (for custom backgrounds)

**Aspect Ratios:**
- `16:9` - Landscape
- `9:16` - Portrait (mobile)
- `1:1` - Square
- `4:5` - Instagram post
- `5:4` - Portrait

**Returns:** Public URL to the generated video with generation details.

## Quick Start Examples

### Text-to-Speech
```python
# Generate speech with custom voice
result = await text_to_speech(
    text="Hello, welcome to Lica World!",
    voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel
    speed=1.2
)
# Returns: Audio URL and conversion details
```

### Image Generation
```python
# Generate an image
result = await text_to_image(
    prompt="A beautiful sunset over mountains",
    aspect_ratio="16:9"
)
# Returns: Image URL and generation details
```

### Video Processing
```python
# Add subtitles to video
result = await add_subtitles(
    video_url="https://example.com/video.mp4",
    subtitle_type="auto"  # Auto-generate from audio
)
# Returns: Subtitled video URL
```

### Canva Integration
```python
# List available Canva tools
tools = await get_external_server_tools("canva")

# Execute Canva tool
result = await execute_external_tool(
    "canva",
    "create_design",
    {"design_type": "presentation"}
)
```

## Development

### Project Structure
```
mcp-server/
├── server.py              # Main MCP server implementation
├── shotstack_helper.py    # Shotstack API integration for subtitles
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### Development Setup

#### Using Docker (Recommended)

1. **Clone and setup:**
```bash
git clone <repository-url>
cd mcp-server
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
docker build -t mcp-server .
docker run -d --name mcp-server-dev -p 8000:8000 --env-file .env -v $(pwd):/app mcp-server

```

#### Manual Setup (Alternative)

1. **Clone and setup:**
```bash
git clone <repository-url>
cd mcp-server
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
python server.py
```

### Adding New AI Tools

1. **Create the tool function:**
```python
@mcp.tool()
async def new_ai_tool(param1: str, param2: int = 10) -> str:
    """Description of what the tool does.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2 (default: 10)
    
    Returns:
        Description of return value
    """
    try:
        # Get API key
        api_key = get_api_key()
        if not api_key:
            return "Error: API key required but not found"
        
        # Initialize client
        client = AIClient(api_key=api_key)
        
        # Process request
        result = await client.process(param1, param2)
        
        # Save file if needed
        filename = f"output_{uuid.uuid4().hex[:8]}.ext"
        file_path = audio_files_dir / filename
        # ... save file logic
        
        # Return public URL
        public_url = f"https://mcp-server.lica.world/endpoint/{filename}"
        return f"Processing successful!\n\nResult URL: {public_url}"
        
    except Exception as e:
        return f"Error: {str(e)}"
```

2. **Add API key helper:**
```python
def get_new_api_key() -> Optional[str]:
    """Get the new API key from environment variables."""
    return os.getenv("NEW_API_KEY")
```

3. **Update environment variables section in README**

### Adding New MCP Servers

```python
# In MCPRegistry.__init__()
self._servers = {
    "canva": {
        "command": "npx",
        "args": ["-y", "mcp-remote@latest", "https://mcp.canva.com/mcp"],
        "description": "Canva design platform integration",
    },
    "new_service": {
        "command": "npx",
        "args": ["-y", "mcp-remote@latest", "https://new-service.mcp.com"],
        "description": "New service integration",
    }
}
```

### Creating Custom Middleware

```python
class CustomMiddleware(Middleware):
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        # Pre-processing
        logger.info(f"Calling tool: {context.tool_name}")
        
        # Execute the tool
        result = await call_next(context)
        
        # Post-processing
        logger.info(f"Tool completed: {context.tool_name}")
        return result

# Register middleware
mcp.add_middleware(CustomMiddleware())
```

### File Serving

Add new file serving endpoints:
```python
@mcp.custom_route("/custom/{filename}", methods=["GET"])
async def serve_custom_files(request: Request) -> FileResponse:
    """Serve custom file types."""
    filename = request.path_params["filename"]
    file_path = audio_files_dir / filename
    
    if not file_path.exists():
        return PlainTextResponse("File not found", status_code=404)
    
    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"inline; filename={filename}"},
    )
```

### Testing Tools

1. **Test individual tools:**
```python
# Test text-to-speech
result = await text_to_speech("Hello world", voice_id="21m00Tcm4TlvDq8ikWAM")
print(result)

# Test image generation
result = await text_to_image("A beautiful sunset", aspect_ratio="16:9")
print(result)
```

2. **Test MCP integration:**
```python
# List external tools
tools = await get_external_server_tools("canva")
print(tools)

# Execute external tool
result = await execute_external_tool("canva", "create_design", {"type": "presentation"})
print(result)
```

### Error Handling Best Practices

1. **API Key Validation:**
```python
def get_api_key() -> Optional[str]:
    """Get API key with validation."""
    api_key = os.getenv("API_KEY")
    if not api_key:
        logger.warning("API key not found in environment")
    return api_key
```

2. **Timeout Management:**
```python
# For long-running operations
start_time = datetime.now()
max_wait_time = 300  # 5 minutes

while True:
    # Check status
    if status == "complete":
        break
    
    elapsed = (datetime.now() - start_time).total_seconds()
    if elapsed > max_wait_time:
        return f"Operation timed out after {max_wait_time} seconds"
    
    await asyncio.sleep(10)  # Wait before checking again
```

3. **File Cleanup:**
```python
try:
    # Process file
    result = process_file(input_path)
    return result
except Exception as e:
    # Clean up temporary files
    if input_path.exists():
        input_path.unlink()
    raise e
```

### Logging

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

# Use in tools
logger.info(f"Starting {tool_name} with params: {params}")
logger.error(f"Error in {tool_name}: {str(e)}")
```

### Performance Optimization

1. **Caching:**
```python
# Cache expensive operations
@lru_cache(maxsize=128)
def expensive_operation(param):
    return compute_expensive_result(param)
```

2. **Async Operations:**
```python
# Use async for I/O operations
async def download_file(url: str) -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()
```

### Debugging

1. **Enable debug logging:**
```python
logging.basicConfig(level=logging.DEBUG)
```

2. **Test with curl:**
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test MCP endpoint
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"method": "tools/list"}'
```

3. **Check file serving:**
```bash
# Test audio file serving
curl http://localhost:8000/audio/test.mp3

# Test image file serving
curl http://localhost:8000/image/test.png
```

## API Endpoints

### Health Check
- **GET** `/health` - Returns "OK" if server is running

### File Serving
- **GET** `/audio/{filename}` - Serve audio files (MP3, WAV, etc.)
- **GET** `/image/{filename}` - Serve image files (PNG, JPG, etc.)
- **GET** `/video/{filename}` - Serve video files (MP4, MOV, etc.)

### MCP Protocol
- **POST** `/mcp` - Main MCP protocol endpoint

## Troubleshooting

### Common Issues

1. **API Key Errors**: Verify all required environment variables are set
2. **File Not Found**: Check if generated files exist in temp directory
3. **Timeout Errors**: Some AI operations take time; check timeout settings
4. **Memory Issues**: Large files may require more memory; consider file size limits

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python server.py
```

### Health Check
```bash
curl http://localhost:8000/health
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
