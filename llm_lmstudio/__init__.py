import base64
import os
import contextlib
from typing import List, Optional

import httpx
import llm
from openai import AsyncOpenAI, OpenAI, APIConnectionError
from pydantic import Field

# LM Studio defaults
DEFAULT_HOST = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"


def get_client(async_mode: bool = False):
    # Check for legacy/Ollama specific env var, but prefer LMSTUDIO_HOST
    base_url = os.getenv("LMSTUDIO_HOST", DEFAULT_HOST)
    api_key = os.getenv("LMSTUDIO_API_KEY", DEFAULT_API_KEY)

    if async_mode:
        return AsyncOpenAI(base_url=base_url, api_key=api_key)
    return OpenAI(base_url=base_url, api_key=api_key)


@llm.hookimpl
def register_models(register):
    """
    Discover models currently loaded in LM Studio.
    Also registers a generic 'lmstudio' model that targets whatever is loaded.
    """
    client = get_client()
    try:
        models_list = client.models.list()
        loaded_models = models_list.data
    except (APIConnectionError, httpx.ConnectError):
        loaded_models = []

    for model_data in loaded_models:
        register(
            LMStudioModel(model_data.id),
            LMStudioAsyncModel(model_data.id),
        )

    # 2. Register a generic "pass-through" model, i.e 'llm -m lmstudio'
    register(
        LMStudioModel("lmstudio", actual_model_id="local-model"),
        LMStudioAsyncModel("lmstudio", actual_model_id="local-model"),
        aliases=["local", "lms"],
    )


@llm.hookimpl
def register_embedding_models(register):
    client = get_client()
    try:
        models_list = client.models.list()
        for model_data in models_list.data:
            register(LMStudioEmbed(model_data.id))
    except Exception:
        pass

    register(LMStudioEmbed("lmstudio", actual_model_id="local-model"))


class _SharedLMStudio:
    can_stream: bool = True
    supports_schema: bool = True
    supports_tools: bool = True

    attachment_types = {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
    }

    class Options(llm.Options):
        """Standard OpenAI-compatible options."""

        temperature: Optional[float] = Field(
            default=None, description="Sampling temperature"
        )
        max_tokens: Optional[int] = Field(
            default=None, description="Max tokens to generate"
        )
        top_p: Optional[float] = Field(
            default=None, description="Nucleus sampling probability"
        )
        frequency_penalty: Optional[float] = Field(
            default=None, description="Frequency penalty"
        )
        presence_penalty: Optional[float] = Field(
            default=None, description="Presence penalty"
        )
        stop: Optional[List[str]] = Field(default=None, description="Stop sequences")

    def __init__(self, model_id: str, actual_model_id: Optional[str] = None):
        self.model_id = model_id
        self.actual_model_id = actual_model_id or model_id

    def __str__(self) -> str:
        return f"LM Studio: {self.model_id}"

    def build_messages(self, prompt, conversation):
        messages = []

        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})

        if conversation:
            for prev_response in conversation.responses:
                # reconstruct structured content if vision model
                content = prev_response.prompt.prompt
                if prev_response.prompt.attachments:
                    content = self._build_content_with_images(
                        prev_response.prompt.prompt, prev_response.prompt.attachments
                    )

                messages.append({"role": "user", "content": content})
                messages.append({"role": "assistant", "content": prev_response.text()})

        content = prompt.prompt
        if prompt.attachments:
            content = self._build_content_with_images(prompt.prompt, prompt.attachments)

        messages.append({"role": "user", "content": content})
        return messages

    def _build_content_with_images(self, text, attachments):
        content_parts = [{"type": "text", "text": text}]
        for attachment in attachments:
            b64 = base64.b64encode(attachment.content).decode("utf-8")
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{attachment.type};base64,{b64}"},
                }
            )
        return content_parts

    def _prepare_tools(self, prompt):
        """Convert LLM tools to OpenAI tool format."""
        if not prompt.tools:
            return None, None

        tools_schema = []
        for tool in prompt.tools:
            tools_schema.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.schema,
                    },
                }
            )
        return tools_schema, "auto"


class LMStudioModel(_SharedLMStudio, llm.Model):
    def execute(self, prompt, stream, response, conversation):
        client = get_client(async_mode=False)
        messages = self.build_messages(prompt, conversation)

        kwargs = prompt.options.model_dump(exclude_none=True)

        # Tool mapping
        tools, tool_choice = self._prepare_tools(prompt)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        # Schema/JSON mode
        if prompt.schema:
            kwargs["response_format"] = {"type": "json_object"}
            last_msg = messages[-1]
            instruction = "\nReply with valid JSON matching the schema."
            if isinstance(last_msg["content"], str):
                last_msg["content"] += instruction
            elif isinstance(last_msg["content"], list):
                for part in last_msg["content"]:
                    if part["type"] == "text":
                        part["text"] += instruction
                        break

        try:
            if stream:
                completion = client.chat.completions.create(
                    model=self.actual_model_id, messages=messages, stream=True, **kwargs
                )
                for chunk in completion:
                    # 2. Token Usage Tracking (Stream)
                    # Check for usage in the final chunk or extra fields
                    if getattr(chunk, "usage", None):
                        response.set_usage(
                            input=chunk.usage.prompt_tokens,
                            output=chunk.usage.completion_tokens,
                        )

                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content

            else:
                completion = client.chat.completions.create(
                    model=self.actual_model_id,
                    messages=messages,
                    stream=False,
                    **kwargs,
                )
                if completion.usage:
                    response.set_usage(
                        input=completion.usage.prompt_tokens,
                        output=completion.usage.completion_tokens,
                    )

                choice = completion.choices[0]
                yield choice.message.content

                # Handle Tool Calls
                if choice.message.tool_calls:
                    for tool_call in choice.message.tool_calls:
                        response.add_tool_call(
                            llm.ToolCall(
                                name=tool_call.function.name,
                                arguments=tool_call.function.arguments,
                            )
                        )

        except (APIConnectionError, httpx.ConnectError) as e:
            raise llm.ModelError(
                f"Could not connect to LM Studio at {client.base_url}.\n"
                "Ensure the server is started (click '<->' -> 'Start Server')."
            ) from e


class LMStudioAsyncModel(_SharedLMStudio, llm.AsyncModel):
    async def execute(self, prompt, stream, response, conversation):
        client = get_client(async_mode=True)
        messages = self.build_messages(prompt, conversation)
        kwargs = prompt.options.model_dump(exclude_none=True)

        tools, tool_choice = self._prepare_tools(prompt)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        if prompt.schema:
            kwargs["response_format"] = {"type": "json_object"}
            last_msg = messages[-1]
            if isinstance(last_msg["content"], str):
                last_msg["content"] += "\nReply with valid JSON."

        try:
            if stream:
                completion = await client.chat.completions.create(
                    model=self.actual_model_id, messages=messages, stream=True, **kwargs
                )
                async for chunk in completion:
                    if getattr(chunk, "usage", None):
                        response.set_usage(
                            input=chunk.usage.prompt_tokens,
                            output=chunk.usage.completion_tokens,
                        )
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
            else:
                completion = await client.chat.completions.create(
                    model=self.actual_model_id,
                    messages=messages,
                    stream=False,
                    **kwargs,
                )
                if completion.usage:
                    response.set_usage(
                        input=completion.usage.prompt_tokens,
                        output=completion.usage.completion_tokens,
                    )
                msg = completion.choices[0].message
                yield msg.content

                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        response.add_tool_call(
                            llm.ToolCall(
                                name=tc.function.name, arguments=tc.function.arguments
                            )
                        )

        except (APIConnectionError, httpx.ConnectError) as e:
            raise llm.ModelError(
                f"Could not connect to LM Studio at {client.base_url}.\n"
                "Ensure the server is started."
            ) from e


class LMStudioEmbed(llm.EmbeddingModel):
    supports_text = True
    supports_binary = False

    def __init__(self, model_id, actual_model_id=None):
        self.model_id = model_id
        self.actual_model_id = actual_model_id or model_id

    def __str__(self) -> str:
        return f"LM Studio: {self.model_id}"

    def embed_batch(self, items):
        client = get_client(async_mode=False)
        try:
            response = client.embeddings.create(
                input=list(items), model=self.actual_model_id
            )
            results = sorted(response.data, key=lambda x: x.index)
            for item in results:
                yield item.embedding
        except (APIConnectionError, httpx.ConnectError) as e:
            raise llm.ModelError("Could not connect to LM Studio server.") from e
