import asyncio
import base64
import glob
import hashlib
import json
import os
import secrets
import uuid
from typing import Any, Dict, Optional
from urllib.parse import urlencode, urlsplit

import requests
import streamlit as st
from assistant import LicaMCPAssistant


def _auth_base_from_resource(resource: str) -> str:
    u = urlsplit(resource)
    return f"{u.scheme}://{u.netloc}"


CANVA_MCP_RESOURCE = "https://mcp.canva.com/mcp"
CANVA_MCP_AUTH_BASE = _auth_base_from_resource(CANVA_MCP_RESOURCE)
OAUTH_REDIRECT_URI = "http://localhost:8501"


# Session Management
def _get_session_id():
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


def _get_pkce_file():
    return f"auth_data/pkce_{_get_session_id()}.json"


def _get_client_file():
    return f"auth_data/client_{_get_session_id()}.json"


# File Management
def _ensure_auth_data_dir():
    """Ensure the auth_data directory exists"""
    os.makedirs("auth_data", exist_ok=True)


def _cleanup_session_files(session_id: str):
    """Clean up all files associated with a specific session"""
    pkce_file = f"auth_data/pkce_{session_id}.json"
    client_file = f"auth_data/client_{session_id}.json"

    for file_path in [pkce_file, client_file]:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _save_json(path: str, data: Dict[str, Any]):
    _ensure_auth_data_dir()
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# Token Management
def _get_client() -> Optional[Dict[str, Any]]:
    return _load_json(_get_client_file())


def _set_client(client_data: Dict[str, Any]):
    _save_json(_get_client_file(), client_data)


def _get_session_tokens() -> Optional[Dict[str, Any]]:
    return st.session_state.get("canva_tokens")


def _set_session_tokens(tokens: Dict[str, Any]):
    st.session_state["canva_tokens"] = tokens


def _clear_session_tokens():
    """Clear session tokens only"""
    if "canva_tokens" in st.session_state:
        del st.session_state["canva_tokens"]


# OAuth Functions
def _gen_pkce() -> Dict[str, str]:
    v = secrets.token_urlsafe(64)
    c = (
        base64.urlsafe_b64encode(hashlib.sha256(v.encode()).digest())
        .decode()
        .rstrip("=")
    )
    state = secrets.token_urlsafe(24)
    return {"verifier": v, "challenge": c, "state": state}


def _discover_auth_server() -> Dict[str, str]:
    url = f"{CANVA_MCP_AUTH_BASE}/.well-known/oauth-authorization-server"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    m = r.json()
    return {
        "authorization_endpoint": m["authorization_endpoint"],
        "token_endpoint": m["token_endpoint"],
        "registration_endpoint": m.get("registration_endpoint"),
    }


def _ensure_client(meta: Dict[str, str]) -> Dict[str, Any]:
    stored = _get_client()
    if stored and stored.get("client_id"):
        return stored

    reg = meta.get("registration_endpoint")
    if reg:
        r = requests.post(
            reg,
            json={
                "redirect_uris": [OAUTH_REDIRECT_URI],
                "token_endpoint_auth_method": "none",
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "client_name": "Lica Design MCP Streamlit Client",
            },
            timeout=10,
        )
        r.raise_for_status()
        info = r.json()
        _set_client(info)
        return info

    cid = os.environ.get("MCP_CLIENT_ID")
    csec = os.environ.get("MCP_CLIENT_SECRET")
    info = {"client_id": cid or f"public-{secrets.token_urlsafe(12)}"}
    if csec:
        info["client_secret"] = csec
    _set_client(info)
    return info


def _build_auth_url(
    meta: Dict[str, str], client_id: str, code_challenge: str, state: str
) -> str:
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": OAUTH_REDIRECT_URI,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "resource": CANVA_MCP_RESOURCE,  # MCP requires resource targeting the protected endpoint
        "state": state,
    }
    return f"{meta['authorization_endpoint']}?{urlencode(params)}"


def _exchange_code(
    meta: Dict[str, str],
    client_id: str,
    code: str,
    code_verifier: str,
    client_secret: Optional[str] = None,
):
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": OAUTH_REDIRECT_URI,
        "client_id": client_id,
        "code_verifier": code_verifier,
        "resource": CANVA_MCP_RESOURCE,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    auth = (client_id, client_secret) if client_secret else None
    r = requests.post(
        meta["token_endpoint"], data=data, headers=headers, auth=auth, timeout=15
    )
    r.raise_for_status()
    return r.json()


def _initiate_oauth():
    meta = _discover_auth_server()
    client = _ensure_client(meta)
    pkce = _gen_pkce()

    session_id = _get_session_id()
    pkce["session_id"] = session_id
    _save_json(_get_pkce_file(), pkce)

    auth_url = _build_auth_url(
        meta, client["client_id"], pkce["challenge"], pkce["state"]
    )
    return auth_url


def _handle_oauth_callback(code: str, state: str):
    pkce = {}

    for file_path in glob.glob("auth_data/pkce_*.json"):
        temp_pkce = _load_json(file_path)
        if temp_pkce and temp_pkce.get("state") == state:
            pkce = temp_pkce
            break

    if not pkce.get("verifier"):
        raise ValueError("PKCE data not found - OAuth flow may have expired")

    if pkce.get("session_id"):
        st.session_state["session_id"] = pkce["session_id"]

    meta = _discover_auth_server()
    client = _ensure_client(meta)
    tokens = _exchange_code(
        meta,
        client["client_id"],
        code,
        pkce["verifier"],
        client.get("client_secret"),
    )
    return tokens


def main():
    st.set_page_config(
        page_title="Lica Design MCP Assistant",
        page_icon="https://cdn.prod.website-files.com/682cc6646b05dce8bde75930/682cc6646b05dce8bde75b66_Lica-min.png",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Add decorative background blur elements
    st.markdown(
        """
    <style>
    .blur-element {
        position: fixed;
        top: -681px;
        width: 80%;
        height: 909px;
        filter: blur(250px);
        z-index: 0;
        pointer-events: none;
    }

    .blur-element-1 {
        right: -25%;
        border-radius: 1428px;
        background: #E6BEFF;
    }
    
    .blur-element-2 {
        left: -25%;
        border-radius: 1222px;
        background: #A7D9FF;
    }
    
    /* Streamlit Header Toolbar Customization - Updated selectors */
    header[data-testid="stHeader"] {
        background-color: rgba(255,255,255,0.3) !important;
        backdrop-filter: blur(20px);
        color: black !important;
    }
    
    .stChatInput > div {
        border: 1px solid #e0e0e0 !important;
        padding: 8px 16px !important;
    }

    .stChatMessage > div[data-testid="stChatMessageAvatarCustom"] {
       border: none !important;
       background-color: transparent !important;
    }
   
    </style>
    <div class="blur-element blur-element-1"></div>
    <div class="blur-element blur-element-2"></div>
    """,
        unsafe_allow_html=True,
    )

    # Handle OAuth callback
    query_params = st.query_params
    if query_params.get("code") and not _get_session_tokens():
        tokens = _handle_oauth_callback(
            query_params.get("code"), query_params.get("state", "")
        )
        _set_session_tokens(tokens)
        if "canva_auth_url" in st.session_state:
            del st.session_state["canva_auth_url"]
        st.query_params.clear()
        # Clean up auth files after successful OAuth
        session_id = _get_session_id()
        _cleanup_session_files(session_id)
        st.rerun()

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        st.divider()

        # Canva connection status
        tokens = _get_session_tokens()
        has_canva_token = tokens and tokens.get("access_token")

        if has_canva_token:
            st.success("Canva Connected")
            if st.button(
                "Disconnect Canva", type="secondary", use_container_width=True
            ):
                _clear_session_tokens()
                if "canva_auth_url" in st.session_state:
                    del st.session_state["canva_auth_url"]
                st.rerun()
        else:
            st.error("Canva not connected")
            if "canva_auth_url" not in st.session_state:
                auth_url = _initiate_oauth()
                st.session_state["canva_auth_url"] = auth_url
            else:
                auth_url = st.session_state["canva_auth_url"]

            # Check if running on localhost
            def is_localhost():
                return st.context.url.startswith(
                    "http://localhost"
                ) or st.context.url.startswith("http://0.0.0.0")

            is_local = is_localhost()

            st.link_button(
                "Connect to Canva",
                auth_url,
                use_container_width=True,
                type="primary",
                disabled=not is_local,
            )
            st.markdown(
                '<p style="text-align: right; font-size: 0.8em; color: #666; font-weight: medium;">*only works on local</p>',
                unsafe_allow_html=True,
            )

        # Available Tools Section
        st.divider()

        # Design & AI Tools
        with st.expander("Available Tools", expanded=False):
            st.markdown(
                """
            **Text & Speech:**
            • Text-to-Speech - Convert text to audio
            • Audio/Video Dubbing - Translate content to other languages
            
            **Image Processing:**
            • Background Removal - Remove image backgrounds
            • Text-to-Image - Generate images from text prompts
            
            **Video Creation:**
            • Text-to-Video - Create videos with AI avatars
            • Image-to-Video - Convert images to animated videos
            • Lip Sync - Sync lip movements with audio
            • Add Subtitles - Add subtitles with auto-transcription
            
            **Design Platform:**
            • Canva Integration - Access Canva design tools
            """
            )

    # Initialize Lica MCP Assistant
    if "mcp_assistant" not in st.session_state:
        st.session_state.mcp_assistant = LicaMCPAssistant()

    # Test MCP server connection
    if "mcp_initialized" not in st.session_state:
        with st.spinner("Connecting to MCP server..."):
            try:
                # Test connection by discovering tools
                tools = asyncio.run(st.session_state.mcp_assistant.discover_tools())
                st.session_state.mcp_initialized = len(tools) > 0
            except Exception as e:
                st.session_state.mcp_initialized = False
                st.error(f"❌ Failed to connect to MCP server: {str(e)}")

    if not st.session_state.mcp_initialized:
        st.error("❌ Failed to connect to MCP server")
        return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Main content area with title and clear chat button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("Lica Design MCP Assistant")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add some vertical spacing
        if st.button("Clear Chat", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input(
        "Ask me anything about design or use the available MCP tools..."
    ):

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user", avatar=":material/account_circle:"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message(
            "assistant",
            avatar="https://cdn.prod.website-files.com/682cc6646b05dce8bde75930/682cc6646b05dce8bde75b66_Lica-min.png",
        ):
            with st.spinner("Thinking..."):
                # Prepare auth headers for Canva
                auth_headers = {}
                tokens = _get_session_tokens()
                if tokens and tokens.get("access_token"):
                    auth_headers["canva-auth-token"] = tokens["access_token"]

                response = asyncio.run(
                    st.session_state.mcp_assistant.handle_user_request(
                        prompt, auth_headers
                    )
                )
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
