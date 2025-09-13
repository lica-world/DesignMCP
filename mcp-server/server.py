"""
Lica MCP Server

This module provides a comprehensive MCP server with various tools including text-to-speech, text-to-video, and Canva integration capabilities.
"""

# Standard library imports
import asyncio
import base64
import io
import logging
import os
import sys
import tempfile
import time
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import jwt as pyjwt

# Third-party imports
import numpy as np
import requests
from dotenv import load_dotenv
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from fastmcp import Client, FastMCP
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import Middleware, MiddlewareContext
from PIL import Image, ImageOps

# Local imports
from shotstack_helper import ElevenLabsTranscriber, ShotstackSubtitleProcessor
from starlette.requests import Request
from starlette.responses import FileResponse, PlainTextResponse

# Load environment variables
load_dotenv()


# ============================================================================
# Constants
# ============================================================================

# Video generation settings
POLLING_INTERVAL = 10
MAX_WAIT_TIME = 300

# Server configuration
TEMP_HOME_PREFIX = "mcp-home-"
AUDIO_FILES_DIR_NAME = "mcp_audio_files"
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"


# ============================================================================
# Configuration
# ============================================================================

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("lica")

# Create temporary directories
temp_home = tempfile.mkdtemp(prefix=TEMP_HOME_PREFIX)
audio_files_dir = Path(tempfile.gettempdir()) / AUDIO_FILES_DIR_NAME
audio_files_dir.mkdir(exist_ok=True)

# Context variable for Canva authentication
canva_auth_token: ContextVar[Optional[str]] = ContextVar(
    "canva_auth_token", default=None
)


# ============================================================================
# MCP Registry
# ============================================================================


class MCPRegistry:
    """Registry for managing multiple MCP servers.

    This class handles the registration, caching, and management of various
    MCP (Model Context Protocol) servers, including Canva integration.
    """

    def __init__(self):
        """Initialize the MCP registry with default server configurations."""
        self._clients = {}
        self._tools_cache = {}
        self._servers = {
            "canva": {
                "command": "npx",
                "args": ["-y", "mcp-remote@latest", "https://mcp.canva.com/mcp"],
                "description": "Canva design platform integration",
            }
        }

    async def get_client(self, server_name: str) -> Client:
        """Get or create MCP client for a server.

        Args:
            server_name: Name of the MCP server

        Returns:
            MCP client instance

        Raises:
            ValueError: If server name is not found
        """
        if server_name not in self._servers:
            raise ValueError(f"Unknown MCP server: {server_name}")

        # Special handling for Canva - use StreamableHttpTransport
        if server_name == "canva":
            token = canva_auth_token.get()
            if not token:
                raise ValueError("Canva access token required but not found")
            return CanvaMCPClient(token)

        # For other servers, use the original MCP client approach
        server_config = self._servers[server_name]
        args = server_config["args"].copy()
        env = {"HOME": temp_home}

        transport = StdioTransport(
            command=server_config["command"],
            args=args,
            env=env,
        )
        return Client(transport)

    async def get_tools(self, server_name: str) -> List[Any]:
        """Get tools for a specific MCP server.

        Args:
            server_name: Name of the MCP server

        Returns:
            List of available tools
        """
        if server_name not in self._tools_cache:
            client = await self.get_client(server_name)

            if server_name == "canva":
                tools_response = await client.list_tools()
                self._tools_cache[server_name] = tools_response
            else:
                async with client:
                    await client.ping()
                    tools_response = await client.list_tools()
                    if hasattr(tools_response, "tools"):
                        self._tools_cache[server_name] = tools_response.tools
                    elif isinstance(tools_response, list):
                        self._tools_cache[server_name] = tools_response
                    else:
                        self._tools_cache[server_name] = getattr(
                            tools_response, "tools", []
                        )
        return self._tools_cache[server_name]

    async def call_tool(
        self, server_name: str, tool_name: str, parameters: Dict[str, Any]
    ) -> Any:
        """Call a tool on a specific MCP server.

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool to call
            parameters: Parameters for the tool

        Returns:
            Tool execution result
        """
        client = await self.get_client(server_name)

        if server_name == "canva":
            return await client.call_tool(tool_name, parameters)
        else:
            async with client:
                await client.ping()
                return await client.call_tool(tool_name, parameters)

    def list_servers(self) -> List[str]:
        """List all registered MCP servers.

        Returns:
            List of server names
        """
        return list(self._servers.keys())

    def add_server(
        self, name: str, command: str, args: List[str], description: str = ""
    ) -> None:
        """Add a new MCP server to the registry.

        Args:
            name: Server name
            command: Command to run the server
            args: Command arguments
            description: Server description
        """
        self._servers[name] = {
            "command": command,
            "args": args,
            "description": description,
        }
        if name in self._tools_cache:
            del self._tools_cache[name]


# ============================================================================
# Canva MCP Client
# ============================================================================


class CanvaMCPClient:
    """MCP client for Canva using FastMCP StreamableHttpTransport.

    This client handles communication with the Canva MCP server using
    HTTP transport with proper authentication headers.
    """

    def __init__(self, access_token: str):
        """Initialize the Canva MCP client with an access token.

        Args:
            access_token: The Canva API access token for authentication.
        """
        self.access_token = access_token
        self.base_url = "https://mcp.canva.com/mcp"
        self._client = None

    def _get_client(self) -> Client:
        """Get or create the MCP client.

        Returns:
            Configured MCP client instance
        """
        if self._client is None:
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": "2025-06-18",
            }
            transport = StreamableHttpTransport(self.base_url, headers=headers)
            self._client = Client(transport)
        return self._client

    async def list_tools(self) -> List[Any]:
        """List available tools from Canva MCP.

        Returns:
            List of available tools
        """
        client = self._get_client()
        async with client:
            tools = await client.list_tools()
            return tools

    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Call a tool on Canva MCP.

        Args:
            tool_name: Name of the tool to call
            parameters: Parameters for the tool

        Returns:
            Tool execution result
        """
        client = self._get_client()
        async with client:
            result = await client.call_tool(tool_name, parameters)
            return result


# ============================================================================
# Middleware
# ============================================================================


class CanvaAuthMiddleware(Middleware):
    """Middleware for extracting and managing Canva authentication tokens.

    This middleware extracts the Canva authentication token from HTTP headers
    and stores it in the context for use by MCP tools.
    """

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        """Extract canva_auth_token from headers and set in context.

        Args:
            context: The middleware context containing request information.
            call_next: The next middleware or tool in the chain.

        Returns:
            The result of the next middleware or tool execution.
        """
        # Retrieve HTTP headers
        headers = get_http_headers()

        # Extract the canva_auth_token
        token = headers.get("canva-auth-token")

        if token:
            # Store the token in the FastMCP context state
            context.fastmcp_context.set_state("canva_auth_token", token)
            canva_auth_token.set(token)
        else:
            canva_auth_token.set(None)

        # Proceed with the next middleware or tool execution
        return await call_next(context)


# ============================================================================
# Global Instances
# ============================================================================

# Global registry
mcp_registry = MCPRegistry()


# ============================================================================
# Utility Functions
# ============================================================================


def get_canva_token() -> Optional[str]:
    """Get the current Canva authentication token from context.

    Returns:
        The Canva authentication token if available, None otherwise.
    """
    return canva_auth_token.get()


def get_elevenlabs_api_key() -> Optional[str]:
    """Get the ElevenLabs API key from environment variables.

    Returns:
        The ElevenLabs API key if available, None otherwise.
    """
    return os.getenv("ELEVENLABS_API_KEY")


def get_synthesia_api_key() -> Optional[str]:
    """Get the Synthesia API key from environment variables.

    Returns:
        The Synthesia API key if available, None otherwise.
    """
    return os.getenv("SYNTHESIA_API_KEY")


def get_background_erase_api_key() -> Optional[str]:
    """Get the Background Erase API key from environment variables.

    Returns:
        The Background Erase API key if available, None otherwise.
    """
    return os.getenv("BACKGROUND_ERASE_API_KEY")


def get_kling_access_key() -> Optional[str]:
    """Get the Kling AI access key from environment variables.

    Returns:
        The Kling AI access key if available, None otherwise.
    """
    return os.getenv("KLING_ACCESS_KEY")


def get_kling_secret_key() -> Optional[str]:
    """Get the Kling AI secret key from environment variables.

    Returns:
        The Kling AI secret key if available, None otherwise.
    """
    return os.getenv("KLING_SECRET_KEY")


def encode_jwt_token(access_key: str, secret_key: str) -> str:
    """Encode JWT token for Kling AI authentication.

    Args:
        access_key: Kling AI access key (used as issuer)
        secret_key: Kling AI secret key (used for signing)

    Returns:
        JWT token string
    """

    headers = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": access_key,  # Your Access Key as issuer
        "exp": int(time.time()) + 1800,  # Token expires in 30 minutes
        "nbf": int(time.time()) - 5,  # Token is valid 5 seconds from now
    }
    return pyjwt.encode(payload, secret_key, headers=headers)


def get_kling_auth_headers() -> Optional[dict]:
    """Get the Kling AI authentication headers with JWT token.

    Returns:
        Dict with JWT authentication headers if keys are available, None otherwise.
    """
    access_key = get_kling_access_key()
    secret_key = get_kling_secret_key()

    if not access_key or not secret_key:
        return None

    # Generate JWT token for authentication
    jwt_token = encode_jwt_token(access_key, secret_key)

    return {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json",
    }


def get_shotstack_api_key() -> Optional[str]:
    """Get the Shotstack API key from environment variables.

    Returns:
        The Shotstack API key if available, None otherwise.
    """
    return os.getenv("SHOTSTACK_API_KEY")


def get_reddit_client_id() -> Optional[str]:
    """Get the Reddit client ID from environment variables.

    Returns:
        The Reddit client ID if available, None otherwise.
    """
    return os.getenv("REDDIT_CLIENT_ID")


def get_reddit_client_secret() -> Optional[str]:
    """Get the Reddit client secret from environment variables.

    Returns:
        The Reddit client secret if available, None otherwise.
    """
    return os.getenv("REDDIT_CLIENT_SECRET")


def get_reddit_user_agent() -> str:
    """Get the Reddit user agent string.

    Returns:
        The Reddit user agent string for API requests.
    """
    return os.getenv("REDDIT_USER_AGENT", "Lica-MCP-Server/1.0 (by /u/LicaBot)")


async def get_reddit_access_token() -> Optional[str]:
    """Get Reddit API access token using client credentials flow.
    
    Returns:
        Access token if successful, None otherwise.
    """
    client_id = get_reddit_client_id()
    client_secret = get_reddit_client_secret()
    user_agent = get_reddit_user_agent()
    
    if not client_id or not client_secret:
        return None
        
    try:
        # Reddit OAuth2 endpoint
        auth_url = "https://www.reddit.com/api/v1/access_token"
        
        # Prepare authentication
        auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
        
        headers = {
            'User-Agent': user_agent,
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'client_credentials'
        }
        
        response = requests.post(auth_url, auth=auth, headers=headers, data=data, timeout=30)
        
        if response.status_code == 200:
            token_data = response.json()
            return token_data.get('access_token')
        else:
            logger.error(f"Failed to get Reddit access token: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting Reddit access token: {str(e)}")
        return None


def parse_reddit_url(url: str) -> Optional[Dict[str, str]]:
    """Parse a Reddit URL to extract subreddit and post information.
    
    Args:
        url: Reddit post URL in various formats
        
    Returns:
        Dictionary with subreddit and post_id, or None if invalid
    """
    import re
    
    # Normalize the URL - remove trailing slashes and query parameters
    url = url.strip().rstrip('/').split('?')[0]
    
    # Various Reddit URL patterns
    patterns = [
        # Standard format: https://www.reddit.com/r/subreddit/comments/post_id/title/
        r'(?:https?://)?(?:www\.|old\.|new\.)?reddit\.com/r/([^/]+)/comments/([^/]+)',
        # Mobile format: https://reddit.com/r/subreddit/comments/post_id/
        r'(?:https?://)?(?:m\.|mobile\.)?reddit\.com/r/([^/]+)/comments/([^/]+)',
        # Short format: https://redd.it/post_id
        r'(?:https?://)?redd\.it/([^/]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                # Standard format with subreddit and post_id
                return {
                    'subreddit': groups[0],
                    'post_id': groups[1]
                }
            elif len(groups) == 1:
                # Short format, need to resolve subreddit via API
                return {
                    'subreddit': None,
                    'post_id': groups[0]
                }
    
    return None


# ============================================================================
# Custom Routes
# ============================================================================


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    """Health check endpoint.

    Args:
        request: The HTTP request object.

    Returns:
        A plain text response indicating the service is healthy.
    """
    return PlainTextResponse("OK")


@mcp.custom_route("/audio/{filename}", methods=["GET"])
async def serve_audio(request: Request) -> FileResponse:
    """Serve audio files.

    Args:
        request: The HTTP request object containing the filename parameter.

    Returns:
        The audio file as a file response, or 404 if not found.
    """
    filename = request.path_params["filename"]
    file_path = audio_files_dir / filename

    if not file_path.exists():
        return PlainTextResponse("File not found", status_code=404)

    return FileResponse(
        file_path,
        media_type="audio/mpeg",
        headers={"Content-Disposition": f"inline; filename={filename}"},
    )


@mcp.custom_route("/image/{filename}", methods=["GET"])
async def serve_image(request: Request) -> FileResponse:
    """Serve image files.

    Args:
        request: The HTTP request object containing the filename parameter.

    Returns:
        The image file as a file response, or 404 if not found.
    """
    filename = request.path_params["filename"]
    file_path = audio_files_dir / filename  # Using same directory for simplicity

    if not file_path.exists():
        return PlainTextResponse("File not found", status_code=404)

    # Determine media type based on file extension
    file_extension = Path(filename).suffix.lower()
    media_type_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
    }
    media_type = media_type_map.get(file_extension, "image/png")

    # Set appropriate headers for transparent images
    headers = {
        "Content-Disposition": f"inline; filename={filename}",
        "Cache-Control": "public, max-age=31536000",  # 1 year cache for static images
    }

    # For PNG files, add headers that help preserve transparency
    if file_extension == ".png":
        headers["X-Content-Type-Options"] = "nosniff"

    return FileResponse(
        file_path,
        media_type=media_type,
        headers=headers,
    )


@mcp.custom_route("/video/{filename}", methods=["GET"])
async def serve_video(request: Request) -> FileResponse:
    """Serve video files.

    Args:
        request: The HTTP request object containing the filename parameter.

    Returns:
        The video file as a file response, or 404 if not found.
    """
    filename = request.path_params["filename"]
    file_path = audio_files_dir / filename  # Using same directory for simplicity

    if not file_path.exists():
        return PlainTextResponse("File not found", status_code=404)

    # Determine media type based on file extension
    file_extension = Path(filename).suffix.lower()
    media_type_map = {
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
        ".flv": "video/x-flv",
        ".wmv": "video/x-ms-wmv",
        ".m4v": "video/x-m4v",
    }
    media_type = media_type_map.get(file_extension, "video/mp4")

    # Set appropriate headers for video files
    headers = {
        "Content-Disposition": f"inline; filename={filename}",
        "Cache-Control": "public, max-age=31536000",  # 1 year cache for static videos
        "Accept-Ranges": "bytes",  # Support for video seeking/streaming
    }

    return FileResponse(
        file_path,
        media_type=media_type,
        headers=headers,
    )


# ============================================================================
# MCP Tools
# ============================================================================


@mcp.tool()
async def get_external_server_tools(server_name: str) -> str:
    """Get all available tools from an external MCP server.

    Args:
        server_name: Name of the external MCP server. Available servers:
            • canva: Canva design platform integration

    Returns:
        A formatted string listing all available tools for the specified external server.
    """
    tools = await mcp_registry.get_tools(server_name)
    result = f"Tools available in external server '{server_name}':\n"
    for tool in tools:
        result += f"  • {tool.name}: {getattr(tool, 'description', 'No description')}\n"
    return result


@mcp.tool()
async def execute_external_tool(
    server_name: str, tool_name: str, parameters: Dict[str, Any]
) -> str:
    """Execute a tool from an external MCP server.

    Args:
        server_name: Name of the external MCP server (e.g., 'canva')
        tool_name: Name of the tool to execute
        parameters: Parameters for the tool (as a dictionary)

    Returns:
        A string indicating the success or failure of the tool execution.
    """
    result = await mcp_registry.call_tool(server_name, tool_name, parameters)
    return f"External tool '{tool_name}' from '{server_name}' executed successfully: {result}"


@mcp.tool()
async def text_to_speech(
    text: str = "Unleash your creativity and explore limitless possibilities with Lica World!",
    voice_id: str = "JBFqnCBsd6RMkjVDRZzb",  # George voice by default
    model_id: str = "eleven_multilingual_v2",
    output_format: str = "mp3_22050_32",
    stability: float = 0.5,
    similarity_boost: float = 0.8,
    style: float = 0.0,
    use_speaker_boost: bool = True,
    speed: float = 1.0,
) -> str:
    """Convert text to speech using ElevenLabs AI and return a public link to the audio file.

    Args:
        text: The text to convert to speech
        voice_id: Voice ID to use
            JBFqnCBsd6RMkjVDRZzb - George (Warm male voice)
            21m00Tcm4TlvDq8ikWAM - Rachel (Natural female voice)
            AZnzlk1XvdvUeBnXmlld - Domi (Confident female voice)
            EXAVITQu4vr4xnSDxMaL - Bella (Expressive female voice)
            ErXwobaYiN019PkySvjV - Antoni (Smooth male voice)
            MF3mGyEYCl7XYWbV9V6O - Elli (Energetic female voice)
            TxGEqnHWrfWFTfGW9XjX - Josh (Deep male voice)
            VR6AewLTigWG4xSOukaG - Arnold (Authoritative male voice)
            pNInz6obpgDQGcFmaJgB - Adam (Professional male voice)
            yoZ06aMxZJJ28mfd3POQ - Sam (Casual male voice)
        model_id: Model to use
            eleven_multilingual_v2
            eleven_monolingual_v1
            eleven_turbo_v2
        output_format: Audio format
            mp3_22050_32
            mp3_44100_32
        stability: Voice stability (0.0-1.0, lower = more variable, higher = more stable)
        similarity_boost: Voice similarity boost (0.0-1.0, higher = more similar to original)
        style: Style exaggeration (0.0-1.0, higher = more exaggerated)
        use_speaker_boost: Whether to use speaker boost for better quality
        speed: Speech speed (0.25-4.0, 1.0 = normal speed)

    Returns:
        A formatted string with the audio URL and conversion details.
    """
    try:
        # Get API key from environment
        api_key = get_elevenlabs_api_key()
        if not api_key:
            return "Error: ElevenLabs API key required but not found. Please set the ELEVENLABS_API_KEY environment variable."

        # Initialize ElevenLabs client
        client = ElevenLabs(api_key=api_key)

        # Configure voice settings
        voice_settings = VoiceSettings(
            stability=stability,
            similarity_boost=similarity_boost,
            style=style,
            use_speaker_boost=use_speaker_boost,
            speed=speed,
        )

        # Generate speech using the convert method
        response = client.text_to_speech.convert(
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
            text=text,
            voice_settings=voice_settings,
        )

        # Generate unique filename
        filename = f"tts_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}.mp3"
        file_path = audio_files_dir / filename

        # Save audio file
        with open(file_path, "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)

        file_size = os.path.getsize(file_path)

        # Return public URL and file info
        public_url = f"https://mcp-server.lica.world/audio/{filename}"

        result = f"Text-to-speech conversion successful!\n\n"
        result += f"Audio URL:   {public_url}\n\n"
        result += f"Text: '{text[:100]}{'...' if len(text) > 100 else ''}'\n"
        result += f"Voice ID: {voice_id}\n"
        result += f"Model: {model_id}\n"
        result += f"Format: {output_format}\n"
        result += f"Speed: {speed}x\n"
        result += f"File size: {file_size:,} bytes\n"

        return result

    except Exception as e:
        return f"Error converting text to speech: {str(e)}"


@mcp.tool()
async def audio_video_dubbing(
    file_url: str,
    target_language: str = "es",  # Spanish by default
    source_language: str = "auto",  # Auto-detect by default
    voice_id: Optional[str] = None,  # Use original voice characteristics if None
    watermark: bool = False,
) -> str:
    """Dub audio or video content to different languages using ElevenLabs AI.

    The output format will match the input format - video files remain as video, audio files as audio.

    Args:
        file_url: URL or path to the audio/video file to dub
        target_language: Target language code
            en - English
            es - Spanish
            fr - French
            de - German
            it - Italian
            pt - Portuguese
            pl - Polish
            tr - Turkish
            ru - Russian
            nl - Dutch
            cs - Czech
            ar - Arabic
            zh - Chinese
            ja - Japanese
            hu - Hungarian
            ko - Korean
        source_language: Source language (auto for auto-detection)
        voice_id: Specific voice ID to use (optional, uses voice cloning if None)
        watermark: Whether to include ElevenLabs watermark

    Returns:
        A formatted string with the dubbed file URL and processing details.
    """
    logger.info(f"Starting dubbing for file: {file_url} to language: {target_language}")
    try:
        # Get API key from environment
        api_key = get_elevenlabs_api_key()
        if not api_key:
            return "Error: ElevenLabs API key required but not found. Please set the ELEVENLABS_API_KEY environment variable."

        # Initialize ElevenLabs client
        client = ElevenLabs(api_key=api_key)

        # Determine file extension from URL or path
        if file_url.startswith(("http://", "https://")):
            # Try to get extension from URL
            original_extension = Path(file_url.split("?")[0]).suffix.lower()
        else:
            original_extension = Path(file_url).suffix.lower()

        # Default to common extensions if not detected
        if not original_extension:
            original_extension = ".mp3"  # Default fallback

        # Download the input file if it's a URL
        input_file_path = None
        if file_url.startswith(("http://", "https://")):
            # Download the file
            response = requests.get(file_url, timeout=60)
            if response.status_code != 200:
                return f"Error: Failed to download file from URL. Status: {response.status_code}"

            # Save temporarily with original extension
            temp_input_filename = f"input_{uuid.uuid4().hex[:8]}{original_extension}"
            input_file_path = audio_files_dir / temp_input_filename

            with open(input_file_path, "wb") as f:
                f.write(response.content)
        else:
            # Assume it's a local file path
            input_file_path = Path(file_url)
            if not input_file_path.exists():
                return f"Error: File not found at path: {file_url}"

        # Start dubbing process
        with open(input_file_path, "rb") as audio_file:
            response = client.dubbing.create(
                file=audio_file,
                target_lang=target_language,
                source_lang=source_language if source_language != "auto" else None,
                num_speakers=1,
                watermark=watermark,
                start_time=None,
                end_time=None,
                highest_resolution=True,
            )

        dubbing_id = (
            response.get("dubbing_id")
            if isinstance(response, dict)
            else getattr(response, "dubbing_id", None)
        )
        if not dubbing_id:
            return f"Error: No dubbing ID returned from ElevenLabs API. Response: {response}"
        logger.info(f"Dubbing started with ID: {dubbing_id}")

        # Poll for completion
        start_time = datetime.now()
        while True:
            # Check dubbing status
            metadata = client.dubbing.get(dubbing_id)
            status = (
                metadata.get("status")
                if isinstance(metadata, dict)
                else getattr(metadata, "status", "unknown")
            )
            logger.info(f"Dubbing status: {status}")

            if status == "dubbed":
                # Dubbing is complete, download the result
                dubbed_stream = client.dubbing.audio.get(dubbing_id, target_language)

                # Generate unique filename for output with original extension
                output_filename = f"dubbed_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}{original_extension}"
                output_file_path = audio_files_dir / output_filename

                # Save dubbed file (preserves original format - audio or video)
                with open(output_file_path, "wb") as f:
                    for chunk in dubbed_stream:
                        if chunk:
                            f.write(chunk)

                file_size = os.path.getsize(output_file_path)
                elapsed_time = (datetime.now() - start_time).total_seconds()

                # Clean up temporary input file if we downloaded it
                if (
                    file_url.startswith(("http://", "https://"))
                    and input_file_path.exists()
                ):
                    input_file_path.unlink()

                # Determine file type for display and serving endpoint
                video_extensions = [
                    ".mp4",
                    ".mov",
                    ".avi",
                    ".mkv",
                    ".webm",
                    ".flv",
                    ".wmv",
                    ".m4v",
                ]
                is_video = original_extension in video_extensions
                file_type = "video" if is_video else "audio"

                # Use appropriate endpoint based on file type
                endpoint = "video" if is_video else "audio"
                public_url = (
                    f"https://mcp-server.lica.world/{endpoint}/{output_filename}"
                )

                result = f"{file_type.title()} dubbing successful!\n\n"
                result += f"Dubbed {file_type.title()} URL: {public_url}\n\n"
                result += f"Original file: {file_url}\n"
                result += f"File type: {file_type.upper()} ({original_extension})\n"
                result += f"Source language: {source_language}\n"
                result += f"Target language: {target_language}\n"
                result += f"Voice ID: {voice_id or 'Voice cloning (original characteristics)'}\n"
                result += f"Processing time: {elapsed_time:.1f} seconds\n"
                result += f"File size: {file_size:,} bytes\n"

                return result

            elif status in ["dubbing", "processing"]:
                # Still processing, wait and check again
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > 600:  # 10 minutes timeout
                    return f"Dubbing timed out after 10 minutes. Dubbing ID: {dubbing_id} (you can check status manually)"

                await asyncio.sleep(10)  # Wait 10 seconds before checking again

            elif status == "error":
                error_message = (
                    metadata.get("error_message")
                    if isinstance(metadata, dict)
                    else getattr(metadata, "error_message", "Unknown error occurred")
                )
                return f"Dubbing failed: {error_message}"

            else:
                # Unknown status, wait and try again
                await asyncio.sleep(10)

    except Exception as e:
        # Clean up temporary file on error
        logger.error(f"Error during audio dubbing: {str(e)}")
        if (
            "input_file_path" in locals()
            and file_url.startswith(("http://", "https://"))
            and input_file_path
            and input_file_path.exists()
        ):
            input_file_path.unlink()
        return f"Error during audio dubbing: {str(e)}"


@mcp.tool()
async def remove_background(
    image_url: str,
) -> str:
    """Remove background from an image using AI background removal.

    Args:
        image_url: URL or path to the image file

    Returns:
        A formatted string with the processed image URL and details.
    """
    logger.info(f"Starting background removal for image: {image_url}")
    output_format = "png"
    try:
        # Get API key from environment
        api_key = get_background_erase_api_key()
        if not api_key:
            return "Error: Background Erase API key required but not found. Please set the BACKGROUND_ERASE_API_KEY environment variable."

        # Download the input image if it's a URL
        input_file_path = None
        if image_url.startswith(("http://", "https://")):
            # Download the image
            response = requests.get(image_url, timeout=60)
            if response.status_code != 200:
                return f"Error: Failed to download image from URL. Status: {response.status_code}"

            # Save temporarily
            temp_input_filename = f"input_{uuid.uuid4().hex[:8]}.jpg"
            input_file_path = audio_files_dir / temp_input_filename

            with open(input_file_path, "wb") as f:
                f.write(response.content)

            # Open the image
            image = Image.open(input_file_path)
        else:
            # Assume it's a local file path
            input_file_path = Path(image_url)
            if not input_file_path.exists():
                return f"Error: Image file not found at path: {image_url}"

            # Open the image
            image = Image.open(input_file_path)

        # Apply EXIF orientation
        image = ImageOps.exif_transpose(image)

        # Check resolution and resize only if needed (over 2K resolution)
        width, height = image.size
        max_dimension = max(width, height)

        if max_dimension > 2000:  # Only resize if larger than 2K resolution
            # Calculate new size maintaining aspect ratio
            if width > height:
                new_width = 1024
                new_height = int((height / width) * 1024)
            else:
                new_height = 1024
                new_width = int((width / height) * 1024)

            image_for_api = image.resize((new_width, new_height), Image.BILINEAR)
            logger.info(
                f"Resized image from {width}x{height} to {new_width}x{new_height}"
            )
        else:
            image_for_api = image
            logger.info(
                f"Image resolution ({width}x{height}) is under 2K, no resizing needed"
            )

        # Convert to base64
        buffer = io.BytesIO()
        image_for_api.save(buffer, format="JPEG", quality=85, optimize=True)
        image_bytes = buffer.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # API request
        headers = {"x-api-key": api_key, "Content-Type": "application/json"}
        payload = {"image": image_base64}

        api_response = requests.post(
            "https://api.backgrounderase.net/v2",
            headers=headers,
            json=payload,
            timeout=60,
        )

        logger.info(
            f"Background removal API response: {api_response}, Status: {api_response.status_code}"
        )

        if api_response.status_code != 200:
            return f"Error: Background removal API failed. Status: {api_response.status_code}, Response: {api_response.text}"

        # Process the response
        result = api_response.json()
        mask_bytes = base64.b64decode(result["mask"])
        mask_img = Image.open(io.BytesIO(mask_bytes))
        mask_array = np.array(mask_img)
        mask = Image.fromarray(mask_array)

        # Apply mask to original image (use full resolution original)
        # Convert to RGBA to support transparency
        original_image = image.convert("RGBA")
        mask_resized = mask.resize(original_image.size, Image.BILINEAR)

        # Ensure mask is in correct mode (L for grayscale alpha)
        if mask_resized.mode != "L":
            mask_resized = mask_resized.convert("L")

        # Create final image with transparency
        final_image = original_image.copy()
        final_image.putalpha(mask_resized)

        # Generate output filename
        output_filename = f"bg_removed_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}.{output_format}"
        output_file_path = audio_files_dir / output_filename

        # Save the final image with proper transparency handling
        if output_format.lower() == "jpg" or output_format.lower() == "jpeg":
            # For JPEG, we need to composite with white background since JPEG doesn't support transparency
            white_background = Image.new("RGB", final_image.size, (255, 255, 255))
            white_background.paste(
                final_image, mask=final_image.split()[-1]
            )  # Use alpha as mask
            white_background.save(
                output_file_path, format="JPEG", quality=95, optimize=True
            )
        else:
            # For PNG/WebP, save with transparency and ensure proper mode
            if final_image.mode != "RGBA":
                final_image = final_image.convert("RGBA")

            # Save with transparency preserved
            save_kwargs = {
                "format": output_format.upper(),
                "optimize": True,
            }

            # PNG-specific options for better transparency
            if output_format.lower() == "png":
                save_kwargs.update(
                    {
                        "compress_level": 6,  # Good balance of speed vs compression
                        "pnginfo": None,  # Don't include metadata that might interfere
                    }
                )

            final_image.save(output_file_path, **save_kwargs)

        file_size = os.path.getsize(output_file_path)

        # Clean up temporary input file if we downloaded it
        if image_url.startswith(("http://", "https://")) and input_file_path.exists():
            input_file_path.unlink()

        # Return public URL and file info
        public_url = f"https://mcp-server.lica.world/image/{output_filename}"

        result = f"Background removal successful!\n\n"
        result += f"Processed Image URL: {public_url}\n\n"
        result += f"Original image: {image_url}\n"
        result += f"Original resolution: {width}x{height}\n"
        result += f"Output format: {output_format.upper()}\n"
        result += f"File size: {file_size:,} bytes\n"

        if max_dimension > 2000:
            result += f"Note: Image was resized for processing but final output maintains original resolution\n"

        return result

    except Exception as e:
        # Clean up temporary file on error
        if (
            "input_file_path" in locals()
            and image_url.startswith(("http://", "https://"))
            and input_file_path
            and input_file_path.exists()
        ):
            input_file_path.unlink()
        return f"Error during background removal: {str(e)}"


@mcp.tool()
async def text_to_image(
    prompt: str,
    aspect_ratio: str = "1:1",
    image_count: int = 1,
    model: str = "kling-v1",
) -> str:
    """Generate images from text descriptions using Kling AI.

    Args:
        prompt: Text description of the image to generate
        aspect_ratio: Aspect ratio for the generated image
            1:1 - Square (1024x1024)
            16:9 - Landscape (1344x768)
            9:16 - Portrait (768x1344)
            4:3 - Standard (1152x896)
            3:4 - Vertical (896x1152)
        image_count: Number of images to generate (1-4, but we'll process one at a time)
        model: AI model to use for generation

    Returns:
        A formatted string with the generated image URL and details.
    """
    logger.info(f"Starting image generation with prompt: {prompt[:100]}...")
    try:
        # Kling API endpoint
        api_url = "https://api.klingai.com/v1/images/generations"

        # Prepare the request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "image_count": 1,  # Generate one image at a time as requested
            "aspect_ratio": aspect_ratio,
        }

        # Get JWT authentication headers
        headers = get_kling_auth_headers()
        if not headers:
            return "Error: Kling AI access and secret keys required but not found. Please set KLING_ACCESS_KEY and KLING_SECRET_KEY environment variables."

        # Submit the generation request
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        logger.info(f"Kling API response: {response}, Status: {response.status_code}")

        if response.status_code != 200:
            error_msg = f"Failed to create image generation task. Status: {response.status_code}, Response: {response.text}"
            logger.error(error_msg)
            return f"Error creating image: {error_msg}"

        response_data = response.json()
        task_id = response_data.get("data", {}).get("task_id")
        logger.info(f"Image generation task created with ID: {task_id}")

        if not task_id:
            return (
                f"Error: No task ID returned from Kling API. Response: {response_data}"
            )

        # Poll for completion
        status_url = f"https://api.klingai.com/v1/images/generations/{task_id}"
        start_time = datetime.now()
        max_wait_time = 300  # 5 minutes timeout

        while True:
            # Check generation status
            status_response = requests.get(status_url, headers=headers, timeout=30)

            if status_response.status_code != 200:
                return f"Error checking image generation status: {status_response.status_code} - {status_response.text}"

            status_data = status_response.json()
            task_status = status_data.get("data", {}).get("task_status")
            logger.info(f"Image generation status: {task_status}, {status_data}")

            if task_status == "succeed":
                # Image generation completed
                image_results = (
                    status_data.get("data", {}).get("task_result", {}).get("images", [])
                )

                if not image_results:
                    return f"Error: Image generation completed but no images returned. Status: {status_data}"

                # Get the first (and only) generated image
                image_data = image_results[0]
                image_url = image_data.get("url")

                if not image_url:
                    return (
                        f"Error: No image URL in generation result. Data: {image_data}"
                    )

                elapsed_time = (datetime.now() - start_time).total_seconds()

                # Return public URL and generation details
                public_url = image_url
                result = f"Image generation successful!\n\n"
                result += f"Generated Image URL: {public_url}\n\n"
                result += (
                    f"Prompt: '{prompt[:200]}{'...' if len(prompt) > 200 else ''}'\n"
                )
                result += f"Model: {model}\n"
                result += f"Aspect ratio: {aspect_ratio}\n"
                result += f"Processing time: {elapsed_time:.1f} seconds\n"

                logger.info(f"Image generation completed successfully: {public_url}")

                return result

            elif task_status == "failed":
                error_detail = (
                    status_data.get("data", {})
                    .get("task_result", {})
                    .get("error", "Unknown error")
                )
                return f"Image generation failed: {error_detail}"

            elif task_status in ["submitted", "processing"]:
                # Still processing, check if we've exceeded max wait time
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > max_wait_time:
                    return f"Image generation timed out after {max_wait_time} seconds. Task ID: {task_id} (you can check status manually)"

                # Wait before checking again
                await asyncio.sleep(5)

            else:
                return f"Unknown generation status: {task_status}. Full response: {status_data}"

    except requests.exceptions.RequestException as e:
        return f"Error communicating with Kling AI API: {str(e)}"
    except Exception as e:
        logger.error(f"Error during image generation: {str(e)}")
        return f"Error during image generation: {str(e)}"


@mcp.tool()
async def image_to_video(
    image_url: str,
    prompt: str = "",
    aspect_ratio: str = "9:16",
    duration: str = "5",
    model: str = "kling-v1",
    cfg: float = 0.5,
) -> str:
    """Convert an image to video using Kling AI image-to-video generation.

    Args:
        image_url: URL or path to the input image
        prompt: Optional text prompt to guide the video generation
        aspect_ratio: Aspect ratio for the generated video
            16:9 - Landscape (1344x768)
            9:16 - Portrait (768x1344)
            1:1 - Square (1024x1024)
        duration: Video duration in seconds ("5" or "10")
        model: AI model to use for generation (kling-v1)
        cfg: Classifier-free guidance scale (0.0-1.0, higher = more prompt adherence)

    Returns:
        A formatted string with the generated video URL and details.
    """
    logger.info(f"Starting image-to-video generation for image: {image_url}")
    try:
        # Download and process the input image if it's a URL
        input_file_path = None
        if image_url.startswith(("http://", "https://")):
            # Download the image
            response = requests.get(image_url, timeout=60)
            if response.status_code != 200:
                return f"Error: Failed to download image from URL. Status: {response.status_code}"

            # Save temporarily
            temp_input_filename = f"input_{uuid.uuid4().hex[:8]}.jpg"
            input_file_path = audio_files_dir / temp_input_filename

            with open(input_file_path, "wb") as f:
                f.write(response.content)

            # Read for base64 encoding
            image_data = response.content
        else:
            # Assume it's a local file path
            input_file_path = Path(image_url)
            if not input_file_path.exists():
                return f"Error: Image file not found at path: {image_url}"

            # Read the local file
            with open(input_file_path, "rb") as f:
                image_data = f.read()

        # Convert image to base64
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Kling API endpoint for image-to-video
        api_url = "https://api.klingai.com/v1/videos/image2video"

        # Prepare the request payload
        payload = {
            "model": model,
            "image": image_base64,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "cfg": cfg,
        }

        # Get JWT authentication headers
        headers = get_kling_auth_headers()
        if not headers:
            return "Error: Kling AI access and secret keys required but not found. Please set KLING_ACCESS_KEY and KLING_SECRET_KEY environment variables."

        # Submit the generation request
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)

        if response.status_code != 200:
            error_msg = f"Failed to create image-to-video task. Status: {response.status_code}, Response: {response.text}"
            return f"Error creating video: {error_msg}"

        response_data = response.json()
        task_id = response_data.get("data", {}).get("task_id")

        if not task_id:
            return (
                f"Error: No task ID returned from Kling API. Response: {response_data}"
            )

        # Poll for completion
        status_url = f"https://api.klingai.com/v1/videos/image2video/{task_id}"
        start_time = datetime.now()
        max_wait_time = 600  # 10 minutes timeout for video generation

        while True:
            # Check generation status
            status_response = requests.get(status_url, headers=headers, timeout=30)

            if status_response.status_code != 200:
                return f"Error checking video generation status: {status_response.status_code} - {status_response.text}"

            status_data = status_response.json()
            task_status = status_data.get("data", {}).get("task_status")

            if task_status == "succeed":
                # Video generation completed
                video_results = (
                    status_data.get("data", {}).get("task_result", {}).get("videos", [])
                )

                if not video_results:
                    return f"Error: Video generation completed but no videos returned. Status: {status_data}"

                # Get the first (and only) generated video
                video_data = video_results[0]
                video_url = video_data.get("url")

                if not video_url:
                    return (
                        f"Error: No video URL in generation result. Data: {video_data}"
                    )

                # Download the generated video
                video_response = requests.get(
                    video_url, timeout=120
                )  # Longer timeout for video download
                if video_response.status_code != 200:
                    return f"Error downloading generated video: {video_response.status_code}"

                # Save the video locally
                output_filename = f"img2vid_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}.mp4"
                output_file_path = audio_files_dir / output_filename

                with open(output_file_path, "wb") as f:
                    f.write(video_response.content)

                file_size = os.path.getsize(output_file_path)
                elapsed_time = (datetime.now() - start_time).total_seconds()

                # Clean up temporary input file if we downloaded it
                if (
                    image_url.startswith(("http://", "https://"))
                    and input_file_path
                    and input_file_path.exists()
                ):
                    input_file_path.unlink()

                # Return public URL and generation details (use video endpoint)
                public_url = f"https://mcp-server.lica.world/video/{output_filename}"

                result = f"Image-to-video generation successful!\n\n"
                result += f"Generated Video URL: {public_url}\n\n"
                result += f"Source image: {image_url}\n"
                result += (
                    f"Prompt: '{prompt}'\n"
                    if prompt
                    else f"Prompt: None (image-only generation)\n"
                )
                result += f"Model: {model}\n"
                result += f"Aspect ratio: {aspect_ratio}\n"
                result += f"Duration: {duration} seconds\n"
                result += f"CFG scale: {cfg}\n"
                result += f"Processing time: {elapsed_time:.1f} seconds\n"
                result += f"File size: {file_size:,} bytes\n"

                return result

            elif task_status == "failed":
                error_detail = (
                    status_data.get("data", {})
                    .get("task_result", {})
                    .get("error", "Unknown error")
                )
                return f"Video generation failed: {error_detail}"

            elif task_status in ["submitted", "processing"]:
                # Still processing, check if we've exceeded max wait time
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > max_wait_time:
                    return f"Video generation timed out after {max_wait_time} seconds. Task ID: {task_id} (you can check status manually)"

                # Wait before checking again (longer intervals for video)
                await asyncio.sleep(10)

            else:
                return f"Unknown generation status: {task_status}. Full response: {status_data}"

    except Exception as e:
        # Clean up temporary file on error
        if (
            "input_file_path" in locals()
            and image_url.startswith(("http://", "https://"))
            and input_file_path
            and input_file_path.exists()
        ):
            input_file_path.unlink()
        return f"Error during image-to-video generation: {str(e)}"


@mcp.tool()
async def lip_sync(
    video_url: str,
    audio_url: str,
    model: str = "kling-v1",
) -> str:
    """Synchronize lip movements in a video with audio using Kling AI lip sync.

    Args:
        video_url: URL or path to the source video file
        audio_url: URL or path to the audio file to sync with
        model: AI model to use for lip sync generation (kling-v1)

    Returns:
        A formatted string with the lip-synced video URL and details.
    """
    logger.info(f"Starting lip sync for video: {video_url} with audio: {audio_url}")
    try:
        # Download and process the video file
        video_file_path = None
        if video_url.startswith(("http://", "https://")):
            # Download the video
            response = requests.get(video_url, timeout=120)
            if response.status_code != 200:
                return f"Error: Failed to download video from URL. Status: {response.status_code}"

            # Save temporarily
            temp_video_filename = f"input_video_{uuid.uuid4().hex[:8]}.mp4"
            video_file_path = audio_files_dir / temp_video_filename

            with open(video_file_path, "wb") as f:
                f.write(response.content)

            video_data = response.content
        else:
            # Assume it's a local file path
            video_file_path = Path(video_url)
            if not video_file_path.exists():
                return f"Error: Video file not found at path: {video_url}"

            with open(video_file_path, "rb") as f:
                video_data = f.read()

        # Download and process the audio file
        audio_file_path = None
        if audio_url.startswith(("http://", "https://")):
            # Download the audio
            response = requests.get(audio_url, timeout=60)
            if response.status_code != 200:
                return f"Error: Failed to download audio from URL. Status: {response.status_code}"

            # Save temporarily
            temp_audio_filename = f"input_audio_{uuid.uuid4().hex[:8]}.mp3"
            audio_file_path = audio_files_dir / temp_audio_filename

            with open(audio_file_path, "wb") as f:
                f.write(response.content)

            audio_data = response.content
        else:
            # Assume it's a local file path
            audio_file_path = Path(audio_url)
            if not audio_file_path.exists():
                return f"Error: Audio file not found at path: {audio_url}"

            with open(audio_file_path, "rb") as f:
                audio_data = f.read()

        # Convert files to base64
        video_base64 = base64.b64encode(video_data).decode("utf-8")
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        # Kling API endpoint for video lip sync
        api_url = "https://api.klingai.com/v1/videos/lip_sync"

        # Prepare the request payload
        payload = {
            "model": model,
            "video": video_base64,
            "audio": audio_base64,
        }

        # Get JWT authentication headers
        headers = get_kling_auth_headers()
        if not headers:
            return "Error: Kling AI access and secret keys required but not found. Please set KLING_ACCESS_KEY and KLING_SECRET_KEY environment variables."

        # Submit the lip sync request
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)

        if response.status_code != 200:
            error_msg = f"Failed to create lip sync task. Status: {response.status_code}, Response: {response.text}"
            return f"Error creating lip sync: {error_msg}"

        response_data = response.json()
        task_id = response_data.get("data", {}).get("task_id")

        if not task_id:
            return (
                f"Error: No task ID returned from Kling API. Response: {response_data}"
            )

        # Poll for completion
        status_url = f"https://api.klingai.com/v1/videos/lip_sync/{task_id}"
        start_time = datetime.now()
        max_wait_time = 900  # 15 minutes timeout for lip sync processing

        while True:
            # Check processing status
            status_response = requests.get(status_url, headers=headers, timeout=30)

            if status_response.status_code != 200:
                return f"Error checking lip sync status: {status_response.status_code} - {status_response.text}"

            status_data = status_response.json()
            task_status = status_data.get("data", {}).get("task_status")

            if task_status == "succeed":
                # Lip sync completed
                video_results = (
                    status_data.get("data", {}).get("task_result", {}).get("videos", [])
                )

                if not video_results:
                    return f"Error: Lip sync completed but no videos returned. Status: {status_data}"

                # Get the lip-synced video
                video_data = video_results[0]
                synced_video_url = video_data.get("url")

                if not synced_video_url:
                    return f"Error: No video URL in lip sync result. Data: {video_data}"

                # Download the lip-synced video
                video_response = requests.get(
                    synced_video_url, timeout=180
                )  # Extended timeout for larger files
                if video_response.status_code != 200:
                    return f"Error downloading lip-synced video: {video_response.status_code}"

                # Save the lip-synced video locally
                output_filename = f"lipsynced_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}.mp4"
                output_file_path = audio_files_dir / output_filename

                with open(output_file_path, "wb") as f:
                    f.write(video_response.content)

                file_size = os.path.getsize(output_file_path)
                elapsed_time = (datetime.now() - start_time).total_seconds()

                # Clean up temporary files if we downloaded them
                if (
                    video_url.startswith(("http://", "https://"))
                    and video_file_path
                    and video_file_path.exists()
                ):
                    video_file_path.unlink()
                if (
                    audio_url.startswith(("http://", "https://"))
                    and audio_file_path
                    and audio_file_path.exists()
                ):
                    audio_file_path.unlink()

                # Return public URL and processing details (use video endpoint)
                public_url = f"https://mcp-server.lica.world/video/{output_filename}"

                result = f"Lip sync successful!\n\n"
                result += f"Lip-Synced Video URL: {public_url}\n\n"
                result += f"Source video: {video_url}\n"
                result += f"Source audio: {audio_url}\n"
                result += f"Model: {model}\n"
                result += f"Processing time: {elapsed_time:.1f} seconds\n"
                result += f"File size: {file_size:,} bytes\n"

                return result

            elif task_status == "failed":
                error_detail = (
                    status_data.get("data", {})
                    .get("task_result", {})
                    .get("error", "Unknown error")
                )
                return f"Lip sync failed: {error_detail}"

            elif task_status in ["submitted", "processing"]:
                # Still processing, check if we've exceeded max wait time
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > max_wait_time:
                    return f"Lip sync timed out after {max_wait_time} seconds. Task ID: {task_id} (you can check status manually)"

                # Wait before checking again
                await asyncio.sleep(15)  # Longer intervals for lip sync processing

            else:
                return f"Unknown lip sync status: {task_status}. Full response: {status_data}"

    except Exception as e:
        # Clean up temporary files on error
        if (
            "video_file_path" in locals()
            and video_url.startswith(("http://", "https://"))
            and video_file_path
            and video_file_path.exists()
        ):
            video_file_path.unlink()
        if (
            "audio_file_path" in locals()
            and audio_url.startswith(("http://", "https://"))
            and audio_file_path
            and audio_file_path.exists()
        ):
            audio_file_path.unlink()
        return f"Error during lip sync: {str(e)}"


@mcp.tool()
async def add_subtitles(
    video_url: str,
) -> str:
    """Add subtitles to a video using Shotstack API with auto-transcription support.

    Args:
        video_url: URL or path to the source video file

    Returns:
        A formatted string with the subtitled video URL and details.
    """
    logger.info(f"Starting subtitle addition for video: {video_url}")
    subtitle_type = "auto"
    subtitles_content = ""
    try:
        # Get API keys from environment
        shotstack_api_key = get_shotstack_api_key()
        if not shotstack_api_key:
            return "Error: Shotstack API key required but not found. Please set the SHOTSTACK_API_KEY environment variable."

        # Initialize Shotstack processor
        processor = ShotstackSubtitleProcessor(shotstack_api_key)

        # Process subtitles based on type
        if subtitle_type.lower() == "auto":
            # Auto-generate subtitles from video audio
            elevenlabs_api_key = get_elevenlabs_api_key()
            if not elevenlabs_api_key:
                return "Error: ElevenLabs API key required for auto transcription. Please set the ELEVENLABS_API_KEY environment variable."

            # Initialize transcriber and generate SRT directly from URL or file
            transcriber = ElevenLabsTranscriber(elevenlabs_api_key)
            srt_content = transcriber.transcribe_video_to_srt(video_url)

            if not srt_content:
                return "Error: Failed to transcribe audio from video. Please check the video has clear audio."

            # Parse the generated SRT content
            subtitles = processor.parse_srt_content(srt_content)
            logger.info(
                f"Auto-generated {len(subtitles)} subtitle segments from video audio"
            )

        elif subtitle_type.lower() == "srt":
            # Parse SRT content
            if not subtitles_content:
                return "Error: SRT content is required when subtitle_type is 'srt'."
            subtitles = processor.parse_srt_content(subtitles_content)

        else:  # text
            # Create subtitles from plain text
            if not subtitles_content:
                return "Error: Text content is required when subtitle_type is 'text'."
            estimated_duration = max(
                30, min(len(subtitles_content) * 0.1, 300)
            )  # 0.1 seconds per character, max 5 minutes
            subtitles = processor.create_subtitles_from_text(
                subtitles_content, estimated_duration
            )

        if not subtitles:
            return "Error: No valid subtitles found in the provided content."

        # Submit render request
        render_id = processor.add_subtitles_to_video(
            video_url=video_url,
            subtitles=subtitles,
        )

        logger.info(f"Shotstack render submitted with ID: {render_id}")

        # Wait for completion
        start_time = datetime.now()
        render_result = processor.wait_for_render_completion(
            render_id=render_id,
            max_wait_time=600,  # 10 minutes
            poll_interval=15,  # Check every 15 seconds
        )

        # Download the rendered video
        rendered_video_url = render_result.get("url")
        logger.info(f"Rendered video URL: {rendered_video_url}, {render_result}")
        if not rendered_video_url:
            return (
                f"Error: No video URL returned from Shotstack. Status: {render_result}"
            )

        elapsed_time = (datetime.now() - start_time).total_seconds()

        # Return public URL and processing details (use video endpoint)
        public_url = rendered_video_url

        result = f"Subtitle addition successful!\n\n"
        result += f"Subtitled Video URL: {public_url}\n\n"
        result += f"Source video: {video_url}\n"
        result += f"Subtitle type: {subtitle_type.upper()}\n"
        if subtitle_type.lower() == "auto":
            result += f"Auto-transcribed from video audio\n"
        result += f"Subtitle count: {len(subtitles)} segments\n"
        result += f"Processing time: {elapsed_time:.1f} seconds\n"
        result += f"Shotstack render ID: {render_id}\n"
        logger.info(f"Subtitle addition result: {result}")
        return result

    except Exception as e:
        logger.error(f"Error during subtitle addition: {str(e)}")
        return f"Error during subtitle addition: {str(e)}"


@mcp.tool()
async def text_to_video(
    text: str = "Unleash your creativity and explore limitless possibilities with Lica World!",
    avatar: str = "anna_costume1_cameraA",
    background: str = "#00A2FF",
    aspect_ratio: str = "9:16",
    title: str = "test video",
) -> str:
    """Convert text to video using Synthesia AI and return a public link to the video file.

    Args:
        text: The text to convert to video (the avatar will speak this text)
        avatar: Avatar to use for the video
            anna_costume1_cameraA - Professional female presenter
            jake_costume1_cameraA - Professional male presenter
            maya_costume1_cameraA - Casual female presenter
            david_costume1_cameraA - Business male presenter
            sarah_costume1_cameraA - Friendly female presenter
        background: Background color or template
            Hex colors: #00A2FF, #FF6B6B, #4ECDC4
            Solid colors: white, black, blue, green, red
            Special: green_screen (for custom backgrounds)
        aspect_ratio: Aspect ratio for the video
            16:9, 9:16, 1:1, 4:5, 5:4
        title: Optional title for the video

    Returns:
        A formatted string with the video URL and generation details.
    """
    polling_interval = POLLING_INTERVAL
    max_wait_time = MAX_WAIT_TIME

    try:
        # Get API key from environment
        api_key = get_synthesia_api_key()
        if not api_key:
            return "Error: Synthesia API key required but not found. Please set the SYNTHESIA_API_KEY environment variable."

        # Synthesia API base URL
        base_url = "https://api.synthesia.io/v2"

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"{api_key}",
        }

        # Create video request payload
        video_data = {
            "test": "false",
            "title": title
            or f"Generated Video - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "visibility": "public",
            "aspectRatio": aspect_ratio,
            "input": [
                {
                    "scriptText": text,
                    "avatar": avatar,
                    "avatarSettings": {
                        "horizontalAlign": "center",
                        "scale": 1,
                        "style": "rectangular",
                        "seamless": False,
                    },
                    "background": (
                        background
                        if background.startswith("#")
                        or background in ["white", "black", "blue", "green", "red"]
                        else "green_screen"
                    ),
                    "backgroundSettings": (
                        {
                            "videoSettings": {
                                "shortBackgroundContentMatchMode": "freeze",
                                "longBackgroundContentMatchMode": "trim",
                            }
                        }
                        if not background.startswith("#")
                        else {}
                    ),
                }
            ],
        }

        response = requests.post(
            f"{base_url}/videos", headers=headers, json=video_data, timeout=30
        )
        if response.status_code != 201:
            error_msg = f"Failed to create video. Status: {response.status_code}, Response: {response.text}"
            return f"Error creating video: {error_msg}"

        video_response = response.json()
        video_id = video_response.get("id")

        if not video_id:
            return f"Error: No video ID returned from Synthesia API. Response: {video_response}"

        # Poll for video completion
        start_time = datetime.now()
        while True:
            # Check video status
            status_response = requests.get(
                f"{base_url}/videos/{video_id}", headers=headers, timeout=30
            )

            if status_response.status_code != 200:
                return f"Error checking video status: {status_response.status_code} - {status_response.text}"

            video_status = status_response.json()
            status = video_status.get("status", "unknown")

            if status == "complete":
                # Video is ready, download it
                download_url = video_status.get("download")

                if not download_url:
                    return f"Error: Video completed but no download URL provided. Status: {video_status}"

                # Return public URL and file info
                public_url = download_url
                elapsed_time = (datetime.now() - start_time).total_seconds()

                result = f"Text-to-video conversion successful!\n\n"
                result += f"Video URL:   {public_url}\n\n"
                result += f"Text: '{text[:100]}{'...' if len(text) > 100 else ''}'\n"
                result += f"Avatar: {avatar}\n"
                result += f"Background: {background}\n"
                result += f"Processing time: {elapsed_time:.1f} seconds\n"
                result += f"Format: MP4\n"

                return result

            elif status == "failed":
                error_detail = video_status.get("error", "Unknown error")
                return f"Video generation failed: {error_detail}"

            elif status in ["in_progress", "queued"]:
                # Check if we've exceeded max wait time
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > max_wait_time:
                    return f"Video generation timed out after {max_wait_time} seconds. Video ID: {video_id} (you can check status manually)"

                await asyncio.sleep(polling_interval)

            else:
                return f"Unknown video status: {status}. Full response: {video_status}"

    except requests.exceptions.RequestException as e:
        return f"Error communicating with Synthesia API: {str(e)}"
    except Exception as e:
        return f"Error converting text to video: {str(e)}"


@mcp.tool()
async def get_reddit_stats(reddit_url: str) -> str:
    """Get comprehensive statistics and information for a Reddit post.

    This tool supports both official Reddit API (with credentials) and public JSON endpoints (fallback).
    For better reliability and higher rate limits, set up Reddit API credentials:
    - REDDIT_CLIENT_ID: Your Reddit app client ID
    - REDDIT_CLIENT_SECRET: Your Reddit app client secret  
    - REDDIT_USER_AGENT: Your app user agent (optional)

    To get Reddit API credentials:
    1. Go to https://www.reddit.com/prefs/apps
    2. Create a new app (script type)
    3. Use the client ID and secret in your environment variables

    Args:
        reddit_url: URL of the Reddit post to analyze
            Supports various formats:
            - https://www.reddit.com/r/subreddit/comments/post_id/title/
            - https://old.reddit.com/r/subreddit/comments/post_id/
            - https://redd.it/post_id
            - https://reddit.com/r/subreddit/comments/post_id/

    Returns:
        A formatted string with comprehensive post statistics and information.
    """
    logger.info(f"Getting Reddit stats for URL: {reddit_url}")
    try:
        # Parse the Reddit URL
        parsed = parse_reddit_url(reddit_url)
        if not parsed:
            return f"Error: Invalid Reddit URL format. Please provide a valid Reddit post URL.\nSupported formats:\n- https://www.reddit.com/r/subreddit/comments/post_id/\n- https://redd.it/post_id"

        subreddit = parsed.get('subreddit')
        post_id = parsed.get('post_id')

        # Try to use official Reddit API first
        access_token = await get_reddit_access_token()
        use_official_api = access_token is not None
        logger.info(f"Using official Reddit API: {use_official_api} {access_token}")
        
        if use_official_api:
            # Use official Reddit API with authentication
            if subreddit:
                api_url = f"https://oauth.reddit.com/r/{subreddit}/comments/{post_id}"
            else:
                api_url = f"https://oauth.reddit.com/comments/{post_id}"
                
            headers = {
                'Authorization': f'Bearer {access_token}',
                'User-Agent': get_reddit_user_agent(),
            }
            logger.info("Using official Reddit API with authentication")
        else:
            # Fall back to public JSON endpoints
            if subreddit:
                api_url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
            else:
                api_url = f"https://www.reddit.com/comments/{post_id}.json"
                
            headers = {
                'User-Agent': get_reddit_user_agent(),
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            logger.info("Using public Reddit JSON endpoints (no authentication)")

        # Make API request
        response = requests.get(api_url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            api_type = "official Reddit API" if use_official_api else "public JSON endpoints"
            return f"Error: Failed to fetch Reddit data from {api_type}. Status: {response.status_code}\nThis could mean the post is private, deleted, or the URL is invalid."

        # Parse JSON response
        data = response.json()
        
        if use_official_api:
            # Official API response format
            if isinstance(data, list) and len(data) > 0:
                # Handle array response format
                post_data = data[0].get('data', {}).get('children', [])
                if not post_data:
                    return f"Error: No post data found in official Reddit API response."
                post = post_data[0].get('data', {})
            else:
                # Handle direct object response
                post = data.get('data', {}) if isinstance(data, dict) else {}
        else:
            # Public JSON endpoint response format  
            if not isinstance(data, list) or len(data) == 0:
                return f"Error: Unexpected Reddit API response format."
                
            post_data = data[0].get('data', {}).get('children', [])
            if not post_data:
                return f"Error: No post data found in Reddit API response."
                
            post = post_data[0].get('data', {})
        
        # Extract comprehensive post information
        title = post.get('title', 'N/A')
        author = post.get('author', 'N/A')
        subreddit_name = post.get('subreddit', 'N/A')
        score = post.get('score', 0)
        upvote_ratio = post.get('upvote_ratio', 0.0)
        num_comments = post.get('num_comments', 0)
        created_utc = post.get('created_utc', 0)
        permalink = post.get('permalink', '')
        
        # Calculate approximate upvotes and downvotes
        if upvote_ratio > 0:
            upvotes = int(score / (2 * upvote_ratio - 1))
            downvotes = upvotes - score
        else:
            upvotes = score
            downvotes = 0
            
        # Format creation date
        from datetime import datetime
        if created_utc:
            created_date = datetime.fromtimestamp(created_utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        else:
            created_date = 'N/A'
            
        # Get post content
        selftext = post.get('selftext', '')
        url = post.get('url', '')
        is_video = post.get('is_video', False)
        is_self = post.get('is_self', False)
        
        # Awards and gilding
        all_awardings = post.get('all_awardings', [])
        total_awards_received = post.get('total_awards_received', 0)
        gilded = post.get('gilded', 0)
        
        # Post flair
        link_flair_text = post.get('link_flair_text', '')
        author_flair_text = post.get('author_flair_text', '')
        
        # Engagement metrics
        num_crossposts = post.get('num_crossposts', 0)
        
        # NSFW and other flags
        over_18 = post.get('over_18', False)
        spoiler = post.get('spoiler', False)
        stickied = post.get('stickied', False)
        locked = post.get('locked', False)
        
        # Build comprehensive result
        result = f"Reddit Post Statistics\n"
        result += f"{'='*50}\n\n"
        
        # Basic Information
        result += f"📝 BASIC INFORMATION\n"
        result += f"Title: {title[:100]}{'...' if len(title) > 100 else ''}\n"
        result += f"Author: u/{author}\n"
        result += f"Subreddit: r/{subreddit_name}\n"
        result += f"Created: {created_date}\n"
        result += f"Post URL: https://reddit.com{permalink}\n\n"
        
        # Engagement Statistics
        result += f"📊 ENGAGEMENT STATISTICS\n"
        result += f"Score (Net Upvotes): {score:,}\n"
        result += f"Upvote Ratio: {upvote_ratio:.1%}\n"
        result += f"Est. Upvotes: ~{upvotes:,}\n"
        result += f"Est. Downvotes: ~{downvotes:,}\n"
        result += f"Comments: {num_comments:,}\n"
        result += f"Crossposts: {num_crossposts:,}\n\n"
        
        # Awards and Recognition
        if total_awards_received > 0 or gilded > 0:
            result += f"🏆 AWARDS & RECOGNITION\n"
            result += f"Total Awards: {total_awards_received:,}\n"
            result += f"Gold/Premium Awards: {gilded}\n"
            
            if all_awardings:
                result += f"Award Breakdown:\n"
                for award in all_awardings[:5]:  # Show top 5 awards
                    award_name = award.get('name', 'Unknown')
                    award_count = award.get('count', 0)
                    result += f"  • {award_name}: {award_count}\n"
                if len(all_awardings) > 5:
                    result += f"  ... and {len(all_awardings) - 5} more award types\n"
            result += "\n"
        
        # Content Information  
        result += f"📄 CONTENT INFORMATION\n"
        result += f"Post Type: "
        if is_self:
            result += "Text Post\n"
        elif is_video:
            result += "Video Post\n"
        elif url and url != reddit_url:
            result += f"Link Post\n"
            result += f"External URL: {url}\n"
        else:
            result += "Media Post\n"
            
        if link_flair_text:
            result += f"Post Flair: {link_flair_text}\n"
        if author_flair_text:
            result += f"Author Flair: {author_flair_text}\n"
            
        # Flags and Status
        flags = []
        if over_18:
            flags.append("NSFW")
        if spoiler:
            flags.append("Spoiler")
        if stickied:
            flags.append("Pinned")
        if locked:
            flags.append("Locked")
            
        if flags:
            result += f"Flags: {', '.join(flags)}\n"
        result += "\n"
        
        # Content Preview (if text post)
        if selftext and len(selftext.strip()) > 0:
            result += f"📖 CONTENT PREVIEW\n"
            preview = selftext[:200].replace('\n', ' ').strip()
            result += f"{preview}{'...' if len(selftext) > 200 else ''}\n\n"
        
        # Engagement Rate Calculation
        if created_utc:
            hours_since_posted = (datetime.now().timestamp() - created_utc) / 3600
            if hours_since_posted > 0:
                engagement_per_hour = (score + num_comments) / hours_since_posted
                result += f"📈 ENGAGEMENT RATE\n"
                result += f"Hours Since Posted: {hours_since_posted:.1f}\n"
                result += f"Engagement per Hour: {engagement_per_hour:.1f} (score + comments)\n\n"
        
        # API source information
        if use_official_api:
            result += f"✅ Data retrieved successfully from official Reddit API (authenticated)\n"
            result += f"📊 Higher rate limits and better reliability available"
        else:
            result += f"✅ Data retrieved from public Reddit JSON endpoints (no auth)\n"
            result += f"💡 For better reliability, set up Reddit API credentials:\n"
            result += f"   - REDDIT_CLIENT_ID: Your app client ID\n"
            result += f"   - REDDIT_CLIENT_SECRET: Your app client secret\n"
            result += f"   - Get credentials at: https://www.reddit.com/prefs/apps"
        
        logger.info(f"Successfully retrieved Reddit stats for post {post_id} using {'official API' if use_official_api else 'public endpoints'}")
        return result
        
    except requests.exceptions.RequestException as e:
        api_type = "official Reddit API" if use_official_api else "public JSON endpoints"
        return f"Error: Network error while fetching Reddit data from {api_type}: {str(e)}"
    except KeyError as e:
        return f"Error: Missing expected data in Reddit API response: {str(e)}"
    except Exception as e:
        logger.error(f"Error getting Reddit stats: {str(e)}")
        return f"Error: Failed to retrieve Reddit post statistics: {str(e)}"


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    """Initialize and run the MCP server."""
    # Register the middleware
    mcp.add_middleware(CanvaAuthMiddleware())

    # Run the server with streamable HTTP transport
    mcp.run(transport="streamable-http", port=DEFAULT_PORT, host=DEFAULT_HOST)
