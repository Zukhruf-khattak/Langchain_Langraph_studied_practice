import inspect
import time
from typing import Any, Callable, Dict, List, Optional

from google import genai
from google.genai import types

from app.config.settings import settings
from app.schemas.chat import ChatMessage, ChatResponse, ToolCallRecord, ToolInfo
from app.tools import AVAILABLE_TOOLS

DEFAULT_SYSTEM_PROMPT = """You are a helpful, intelligent, and accurate AI assistant.
You have access to specialized tools for calculations, date/time lookups, and text analysis.
When a user query requires exact computation, current date/time information, or textual metrics, ALWAYS use the relevant tool to provide an accurate answer.
Explain your reasoning clearly and concisely after obtaining tool outputs."""


class Agent:
    """Autonomous AI Agent that interacts with Gemini LLM and executes registered tools."""

    def __init__(self, tools: Optional[List[Callable[..., Any]]] = None):
        self.tool_functions: Dict[str, Callable[..., Any]] = {}
        for tool_fn in (tools or AVAILABLE_TOOLS):
            self.register_tool(tool_fn)

    def register_tool(self, func: Callable[..., Any]) -> None:
        """Register a python callable tool."""
        self.tool_functions[func.__name__] = func

    def get_registered_tools_info(self) -> List[ToolInfo]:
        """Get schema and docstring information for all registered tools."""
        tool_infos: List[ToolInfo] = []
        for name, func in self.tool_functions.items():
            doc = inspect.getdoc(func) or "No description provided."
            sig = inspect.signature(func)

            param_props: Dict[str, Any] = {}
            required_params: List[str] = []

            for param_name, param in sig.parameters.items():
                param_type = "string"
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"

                param_props[param_name] = {
                    "type": param_type,
                    "default": None if param.default == inspect.Parameter.empty else param.default,
                }
                if param.default == inspect.Parameter.empty:
                    required_params.append(param_name)

            tool_infos.append(
                ToolInfo(
                    name=name,
                    description=doc.split("\n\n")[0] if "\n\n" in doc else doc,
                    parameters={
                        "type": "object",
                        "properties": param_props,
                        "required": required_params,
                    },
                )
            )
        return tool_infos

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a registered tool by name with provided keyword arguments."""
        if tool_name not in self.tool_functions:
            return f"Error: Tool '{tool_name}' not found."
        func = self.tool_functions[tool_name]
        try:
            return func(**arguments)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"

    async def run(
        self,
        message: str,
        history: Optional[List[ChatMessage]] = None,
        system_prompt: Optional[str] = None,
    ) -> ChatResponse:
        """
        Execute the agent loop for a user query.

        Args:
            message: Current user message.
            history: Optional list of previous chat messages.
            system_prompt: Optional system prompt override.

        Returns:
            ChatResponse containing the final reply, list of executed tools, and execution metadata.
        """
        start_time = time.time()
        tool_call_records: List[ToolCallRecord] = []

        if not settings.GEMINI_API_KEY:
            # Helpful mock response when GEMINI_API_KEY is not set yet
            return ChatResponse(
                reply=(
                    "[CONFIG NOTICE] GEMINI_API_KEY is not configured.\n\n"
                    "Please set your GEMINI_API_KEY in the .env file or environment variables to enable the AI agent. "
                    "You can obtain an API key at https://aistudio.google.com/."
                ),
                tool_calls=[],
                model=settings.MODEL_NAME,
                execution_time_seconds=round(time.time() - start_time, 4),
            )

        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        sys_instruction = system_prompt or DEFAULT_SYSTEM_PROMPT

        # Build contents list
        contents: List[types.Content] = []

        # Add history if provided
        if history:
            for item in history:
                role = "user" if item.role in ["user", "human"] else "model"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=item.content)],
                    )
                )

        # Add current user prompt
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=message)],
            )
        )

        # Pass available tool callables directly to Google GenAI config
        tools_list = list(self.tool_functions.values())
        config = types.GenerateContentConfig(
            system_instruction=sys_instruction,
            tools=tools_list,
            temperature=0.2,
        )

        final_reply = ""
        iterations = 0
        max_iterations = settings.MAX_TOOL_ITERATIONS

        while iterations < max_iterations:
            iterations += 1

            response = client.models.generate_content(
                model=settings.MODEL_NAME,
                contents=contents,
                config=config,
            )

            # Check if model made function / tool calls
            has_function_call = False
            if response.function_calls:
                has_function_call = True
                for fc in response.function_calls:
                    fn_name = fc.name
                    fn_args = dict(fc.args) if fc.args else {}

                    # Execute the tool
                    output = self._execute_tool(fn_name, fn_args)

                    # Record the tool call
                    tool_call_records.append(
                        ToolCallRecord(
                            tool_name=fn_name,
                            arguments=fn_args,
                            result=output,
                        )
                    )

                    # Append model's function call message to contents
                    contents.append(
                        types.Content(
                            role="model",
                            parts=[
                                types.Part.from_function_call(
                                    name=fn_name,
                                    args=fn_args,
                                )
                            ],
                        )
                    )

                    # Append function response to contents
                    contents.append(
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_function_response(
                                    name=fn_name,
                                    response={"result": output},
                                )
                            ],
                        )
                    )

            # If model didn't call functions or provided final text
            if not has_function_call:
                final_reply = response.text or ""
                break

        if not final_reply and tool_call_records:
            final_reply = "Agent completed tool executions successfully."

        execution_time = round(time.time() - start_time, 4)

        return ChatResponse(
            reply=final_reply,
            tool_calls=tool_call_records,
            model=settings.MODEL_NAME,
            execution_time_seconds=execution_time,
        )


# Global Agent instance initialized with default tools
agent = Agent()
