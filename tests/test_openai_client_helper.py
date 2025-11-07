import asyncio
import sys
import time
import types
from importlib import import_module


def _make_fake_openai_module(response_obj=None, delay: float = 0):
    """Create a fake openai package with the minimal symbols used by the helper."""
    openai_mod = types.ModuleType("openai")

    class OpenAIError(Exception):
        pass

    # types.chat submodule
    types_mod = types.ModuleType("openai.types")
    chat_mod = types.ModuleType("openai.types.chat")

    class ChatCompletion:
        def __init__(self, choices):
            self.choices = choices

    ChatCompletionMessageParam = dict

    # Fake OpenAI client
    class FakeChatCompletions:
        def __init__(self, response_obj, delay):
            self._response = response_obj
            self._delay = delay

        def create(self, model, messages, **kwargs):
            if self._delay:
                time.sleep(self._delay)
            return self._response

    class FakeChat:
        def __init__(self, response_obj, delay):
            self.completions = FakeChatCompletions(response_obj, delay)

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None, organization=None):
            # store config to inspect in tests
            self.api_key = api_key
            self.base_url = base_url
            self.organization = organization
            self.chat = FakeChat(response_obj, delay)

    # assemble modules
    openai_mod.OpenAI = FakeOpenAI
    err_mod = types.ModuleType("openai.error")
    err_mod.OpenAIError = OpenAIError

    chat_mod.ChatCompletion = ChatCompletion
    chat_mod.ChatCompletionMessageParam = ChatCompletionMessageParam

    types_mod.chat = chat_mod

    sys.modules["openai"] = openai_mod
    sys.modules["openai.error"] = err_mod
    sys.modules["openai.types"] = types_mod
    sys.modules["openai.types.chat"] = chat_mod

    return ChatCompletion


def test_ask_returns_content_and_accepts_return_raw(monkeypatch):
    # prepare fake response object
    ChatCompletion = _make_fake_openai_module()
    msg = types.SimpleNamespace(content="Hello world")
    choice = types.SimpleNamespace(message=msg)
    fake_resp = ChatCompletion([choice])

    # re-insert fake module with response
    _make_fake_openai_module(response_obj=fake_resp, delay=0)

    import importlib as _importlib
    _mod = import_module("src.providers.openai_client_helper")
    _importlib.reload(_mod)
    helper = _mod.OpenAIClientHelper(api_key="x")

    result = asyncio.run(helper.ask([{"role": "user", "content": "hi"}], model="gpt", return_raw=False))
    assert result == "Hello world"

    raw = asyncio.run(helper.ask([{"role": "user", "content": "hi"}], model="gpt", return_raw=True))
    assert isinstance(raw, ChatCompletion)


def test_ask_timeout(monkeypatch):
    ChatCompletion = _make_fake_openai_module()
    msg = types.SimpleNamespace(content="Delayed")
    choice = types.SimpleNamespace(message=msg)
    fake_resp = ChatCompletion([choice])

    # delay the fake client to trigger timeout
    _make_fake_openai_module(response_obj=fake_resp, delay=0.5)

    import importlib as _importlib
    _mod = import_module("src.providers.openai_client_helper")
    _importlib.reload(_mod)
    helper = _mod.OpenAIClientHelper(api_key="x")

    # small timeout to force TimeoutError and get None
    result = asyncio.run(helper.ask([{"role": "user", "content": "hi"}], model="gpt", timeout=0.01))
    assert result is None
