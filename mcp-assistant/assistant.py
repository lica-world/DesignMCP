"""
Lica MCP Assistant

This module provides a streamlined MCP assistant that uses GPT to dynamically
orchestrate tool execution based on user requests and previous results.
"""

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import anthropic
from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

load_dotenv()


# ============================================================================
# Data Models
# ============================================================================


class StepStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    NEEDS_MORE_INFO = "needs_more_info"
    COMPLETED = "completed"


@dataclass
class ExecutionStep:
    """Represents a single execution step with its result."""

    step_number: int
    tool_name: str
    parameters: Dict[str, Any]
    description: str
    status: StepStatus
    result: str = ""
    error: str = ""
    reasoning: str = ""


# ============================================================================
# MCP Connection Handler
# ============================================================================


class MCPConnectionHandler:
    """Handles MCP server connections and basic operations."""

    def __init__(self, server_url: str):
        self.server_url = server_url
        self._get_headers = lambda: {
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-06-18",
        }

    def _create_transport(
        self, auth_headers: Optional[Dict[str, str]] = None
    ) -> StreamableHttpTransport:
        """Create a transport with appropriate headers."""
        headers = self._get_headers()
        if auth_headers:
            headers.update(auth_headers)
        return StreamableHttpTransport(self.server_url, headers=headers)

    async def discover_tools(self) -> List[Dict[str, Any]]:
        """Discover all available tools from the MCP server."""

        transport = self._create_transport()
        client = Client(transport)

        async with client:
            await client.ping()
            response = await client.list_tools()

            if isinstance(response, list):
                tool_list = response
            elif hasattr(response, "tools"):
                tool_list = response.tools
            else:
                tool_list = getattr(response, "tools", [])

            all_tools = []
            for tool in tool_list:
                tool_info = {
                    "name": tool.name if hasattr(tool, "name") else tool["name"],
                    "description": getattr(tool, "description", "")
                    or tool.get("description", ""),
                    "parameters": getattr(tool, "inputSchema", {})
                    or tool.get("inputSchema", {}),
                }
                all_tools.append(tool_info)

            return all_tools

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        auth_headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[StepStatus, str]:
        """Execute a single tool and return the result."""
        transport = self._create_transport(auth_headers)
        client = Client(transport)

        async with client:
            await client.ping()
            result = await client.call_tool(tool_name, parameters)

            if hasattr(result, "content") and result.content:
                result_text = result.content[0].text
            elif hasattr(result, "data"):
                result_text = result.data
            else:
                result_text = str(result)

            return StepStatus.SUCCESS, result_text


# ============================================================================
# GPT Decision Engine
# ============================================================================


class GPTDecisionEngine:
    """Handles GPT-based decision making for tool orchestration."""

    def __init__(self):
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    def _prepare_tools_info(
        self, available_tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prepare tools information for GPT."""
        tools_info = []
        for tool in available_tools:
            tool_info = {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            }
            tools_info.append(tool_info)
        return tools_info

    def _prepare_execution_history(self, execution_history: List[ExecutionStep]) -> str:
        """Format execution history for GPT."""
        if not execution_history:
            return ""

        history_text = "\n\nPREVIOUS STEPS (DO NOT REPEAT THESE):\n"
        for step in execution_history:
            status_emoji = (
                "✅"
                if step.status == StepStatus.SUCCESS
                else "❌" if step.status == StepStatus.FAILED else "⚠️"
            )
            history_text += (
                f"{status_emoji} Step {step.step_number}: {step.tool_name}\n"
            )
            history_text += f"   Description: {step.description}\n"
            history_text += f"   Parameters: {json.dumps(step.parameters, indent=2)}\n"
            if step.result:
                history_text += f"   Result: {step.result}\n"
            if step.error:
                history_text += f"   Error: {step.error}\n"
            history_text += "\n"

        return history_text

    def _load_system_prompt(
        self, tools_info: List[Dict[str, Any]], execution_history: str
    ) -> str:
        """Load and format the system prompt."""
        with open("system_prompt.txt", "r") as f:
            system_prompt_template = f.read()

        return system_prompt_template.format(
            available_tools=json.dumps(tools_info, indent=2),
            execution_history=execution_history,
        )

    def _extract_json_from_response(self, raw_content: str) -> str:
        """Extract JSON content from GPT response."""
        json_content = raw_content.strip()
        if not json_content.startswith("{"):
            start_idx = json_content.find("{")
            end_idx = json_content.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_content = json_content[start_idx : end_idx + 1]
        return json_content

    def _validate_decision(self, result: Dict[str, Any]) -> None:
        """Validate the decision response from GPT."""
        if "action" not in result:
            raise ValueError("Missing 'action' field in response")
        if result["action"] == "call_tool":
            if "tool_name" not in result:
                raise ValueError("Missing 'tool_name' field for call_tool action")
            if "parameters" not in result:
                raise ValueError("Missing 'parameters' field for call_tool action")

    def decide_next_step(
        self,
        user_request: str,
        execution_history: List[ExecutionStep],
        available_tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Use GPT to decide what to do next based on the user request and execution history."""
        tools_info = self._prepare_tools_info(available_tools)
        history_text = self._prepare_execution_history(execution_history)
        system_prompt = self._load_system_prompt(tools_info, history_text)

        response = self.anthropic_client.messages.create(
            model="claude-opus-4-1-20250805",
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"User request: {user_request}",
                }
            ],
            max_tokens=8092,
        )

        json_content = self._extract_json_from_response(response.content[0].text)
        result = json.loads(json_content)
        self._validate_decision(result)

        return result


# ============================================================================
# Execution Orchestrator
# ============================================================================


class ExecutionOrchestrator:
    """Orchestrates the execution of tools based on GPT decisions."""

    def __init__(
        self,
        mcp_handler: MCPConnectionHandler,
        gpt_engine: GPTDecisionEngine,
        max_steps: int = 10,
    ):
        self.mcp_handler = mcp_handler
        self.gpt_engine = gpt_engine
        self.max_steps = max_steps

    def _create_execution_step(
        self,
        step_number: int,
        tool_name: str,
        parameters: Dict[str, Any],
        description: str,
        reasoning: str,
        status: StepStatus,
        result: str = "",
    ) -> ExecutionStep:
        """Create an execution step record."""
        return ExecutionStep(
            step_number=step_number,
            tool_name=tool_name,
            parameters=parameters,
            description=description,
            status=status,
            result=result if status == StepStatus.SUCCESS else "",
            error=result if status == StepStatus.FAILED else "",
            reasoning=reasoning,
        )

    async def _handle_decision_action(
        self,
        decision: Dict[str, Any],
        step_number: int,
        execution_history: List[ExecutionStep],
        auth_headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, str]:
        """Handle different decision actions and return (should_continue, result)."""
        action = decision.get("action")

        if action == "complete":
            return False, decision.get("final_response", "Task completed successfully")

        elif action == "ask_for_info":
            return (
                False,
                f"I need more information to help you: {decision.get('description', 'Please provide more details')}",
            )

        elif action == "call_tool":
            tool_name = decision.get("tool_name")
            parameters = decision.get("parameters", {})
            description = decision.get("description", f"Executing {tool_name}")
            reasoning = decision.get("reasoning", "")

            # Execute the tool
            status, result = await self.mcp_handler.execute_tool(
                tool_name, parameters, auth_headers
            )

            # Record the step
            step = self._create_execution_step(
                step_number=step_number,
                tool_name=tool_name,
                parameters=parameters,
                description=description,
                reasoning=reasoning,
                status=status,
                result=result,
            )
            execution_history.append(step)

            # If the tool failed, stop execution
            if status == StepStatus.FAILED:
                return False, ""

            return True, ""

        return False, "Unknown action"

    async def execute_request(
        self,
        user_request: str,
        available_tools: List[Dict[str, Any]],
        auth_headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """Execute a user request through dynamic tool orchestration."""
        execution_history = []
        step_number = 1

        # Start the dynamic execution loop
        while step_number <= self.max_steps:
            decision = self.gpt_engine.decide_next_step(
                user_request, execution_history, available_tools
            )

            should_continue, result = await self._handle_decision_action(
                decision, step_number, execution_history, auth_headers
            )

            if not should_continue:
                return result

            step_number += 1

        # Max steps reached - ask GPT for a final response
        if execution_history:
            final_decision = self.gpt_engine.decide_next_step(
                f"Generate a final response for the user. Original request: {user_request}. Some steps were completed but execution stopped.",
                execution_history,
                available_tools,
            )
            if final_decision.get("action") == "complete" and final_decision.get(
                "final_response"
            ):
                return final_decision.get("final_response")

        return f"I encountered an issue while processing your request: '{user_request}'. Please try rephrasing your question or ask for something else."


# ============================================================================
# Main Assistant Class
# ============================================================================


class LicaMCPAssistant:
    """
    Lica MCP Assistant that uses GPT to dynamically orchestrate tool execution.

    This approach:
    1. Starts with the user request
    2. Uses GPT to decide what tool to call first
    3. Executes the tool and gets the result
    4. Shows GPT the result and asks what to do next
    5. Repeats until the task is complete or GPT decides to stop
    """

    def __init__(self):
        self.mcp_server_url = os.getenv("LICA_MCP_URL")
        self.available_tools: List[Dict[str, Any]] = []
        self.max_steps = 10  # Maximum GPT decision loops

        # Initialize components
        self.mcp_handler = MCPConnectionHandler(self.mcp_server_url)
        self.gpt_engine = GPTDecisionEngine()
        self.orchestrator = ExecutionOrchestrator(
            self.mcp_handler, self.gpt_engine, self.max_steps
        )

    async def discover_tools(self) -> List[Dict[str, Any]]:
        """
        Discover all available tools from all MCP servers.

        Returns:
            List of available tools from all servers
        """
        self.available_tools = await self.mcp_handler.discover_tools()
        return self.available_tools

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        auth_headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[StepStatus, str]:
        """
        Execute a single tool and return the result.

        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters for the tool
            auth_headers: Optional authentication headers

        Returns:
            Tuple of (status, result)
        """
        return await self.mcp_handler.execute_tool(tool_name, parameters, auth_headers)

    def decide_next_step(
        self,
        user_request: str,
        execution_history: List[ExecutionStep],
        available_tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Use GPT to decide what to do next based on the user request and execution history.

        Args:
            user_request: Original user request
            execution_history: List of previous execution steps
            available_tools: List of available tools

        Returns:
            Dictionary with next action to take
        """
        return self.gpt_engine.decide_next_step(
            user_request, execution_history, available_tools
        )

    async def handle_user_request(
        self, user_request: str, auth_headers: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Handle a user request dynamically by making decisions step by step.

        Args:
            user_request: The user's request
            auth_headers: Optional authentication headers

        Returns:
            Response to the user
        """
        # Discover tools if not already done
        if not self.available_tools:
            await self.discover_tools()

        return await self.orchestrator.execute_request(
            user_request, self.available_tools, auth_headers
        )

    def get_available_tools_summary(self) -> str:
        """
        Get a summary of all available tools.

        Returns:
            Formatted string with tool summary
        """
        if not self.available_tools:
            return "No tools available. Run discover_tools() first."

        summary = "Available MCP tools:\n\n"
        for tool in self.available_tools:
            summary += f"• **{tool['name']}**: {tool['description']}\n"

        return summary
