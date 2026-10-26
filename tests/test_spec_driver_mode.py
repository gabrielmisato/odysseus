import json
import sys
import types
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from src import clean_agent_preview as runner
from src.tool_policy import ToolPolicy
from src.tool_schemas import FUNCTION_TOOL_SCHEMAS
from src.turn_contract import (
    requested_capabilities,
    resolve_turn_contract,
    selected_tools_for_request,
)

PROMPT = 'Search the web for the Ajax population.'


def _contract(prompt=PROMPT):
    policy = ToolPolicy()
    selected = selected_tools_for_request(prompt)
    return resolve_turn_contract(
        capabilities=requested_capabilities(prompt),
        schemas=FUNCTION_TOOL_SCHEMAS,
        policy=policy,
        selected_tools=selected,
        message=prompt,
    ), policy


def _install_driver_router(monkeypatch, decision, picks=None):
    """Fake src.spec_router exposing spec_route_tools + LAST_DECISIONS."""
    fake = types.ModuleType('src.spec_router')
    fake.think_router_enabled = lambda: False
    fake.LAST_DECISIONS = [decision]

    def _spec_route_tools(user_message, schemas, disabled_tools=None,
                          messages=None, return_decision=False):
        if picks is not None:
            chosen = picks
        elif decision.get('tool_name'):
            chosen = [decision['tool_name']]
        else:
            chosen = list(decision.get('tools') or [])
        return (chosen, True) if return_decision else chosen

    fake.spec_route_tools = _spec_route_tools
    monkeypatch.setitem(sys.modules, 'src.spec_router', fake)
    monkeypatch.setattr(
        'src.agent_loop._configured_model_tool_surface_raw',
        lambda *a, **k: 'spec', raising=False,
    )


def _fake_response(captured, scripts):
    pending = list(scripts)

    class Response:
        def __init__(self, script=None):
            self._s = script or {}

        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            calls = self._s.get('tool_calls')
            if calls:
                for i, c in enumerate(calls):
                    yield 'data: ' + json.dumps({'choices': [{'delta': {'tool_calls': [
                        {'index': i, 'id': f'c{i}', 'type': 'function',
                         'function': {'name': c['name'],
                                      'arguments': c.get('arguments') or '{}'}}]}}]})
            else:
                content = self._s.get('content', 'Hi.')
                yield 'data: ' + json.dumps({'choices': [{'delta': {'content': content}}]})
            yield 'data: [DONE]'

    @asynccontextmanager
    async def response(*args, **kwargs):
        captured.append(args[3] if len(args) > 3 else kwargs.get('request'))
        yield Response(pending.pop(0) if pending else None)

    return response


async def _run(monkeypatch, captured, model='odysseus-agentic-v4', scripts=None,
               max_rounds=1):
    monkeypatch.setattr(
        runner, 'preview_model_response', _fake_response(captured, scripts or []))
    contract, policy = _contract()
    events = []
    async for chunk in runner.stream_preview(
        endpoint_url='http://fixture', model=model, headers={},
        messages=[{'role': 'user', 'content': PROMPT}], turn_contract=contract,
        session_id='spec-driver', owner='fixture', disabled_tools=set(),
        tool_policy=policy, thinking_mode='off', max_rounds=max_rounds,
    ):
        if chunk.startswith('data: ') and '[DONE]' not in chunk:
            events.append(json.loads(chunk[6:]))
    return contract, events


def _install_recorder(monkeypatch):
    executed = []

    async def _rec(block, *args, **kwargs):
        executed.append(getattr(block, 'tool_type', None))
        return 'ok', {'exit_code': 0, 'output': 'ok'}

    monkeypatch.setattr(runner, 'execute_tool_block', _rec)
    return executed


def test_spec_driver_active_gate(monkeypatch):
    monkeypatch.setenv('ODYSSEUS_SPEC_MODE', 'driver')
    monkeypatch.setattr(
        'src.agent_loop._configured_model_tool_surface_raw',
        lambda *a, **k: 'spec', raising=False)
    assert runner.spec_driver_active('http://x', 'm', 'o', {}) is True
    monkeypatch.setenv('ODYSSEUS_SPEC_MODE', 'strict')
    assert runner.spec_driver_active('http://x', 'm', 'o', {}) is False
    monkeypatch.setenv('ODYSSEUS_SPEC_MODE', 'driver')
    monkeypatch.setattr(
        'src.agent_loop._configured_model_tool_surface_raw',
        lambda *a, **k: 'compact', raising=False)
    assert runner.spec_driver_active('http://x', 'm', 'o', {}) is False


def test_spec_driver_pick_shapes(monkeypatch):
    monkeypatch.setenv('ODYSSEUS_SPEC_MODE', 'driver')
    # tool pick at any confidence
    _install_driver_router(monkeypatch, {
        'message': PROMPT, 'choice': 't3', 'tools': ['web_fetch'],
        'top': [('web_fetch', 0.12)], 'answer_directly': 0.2, 'ask_user': 0.1,
    })
    pick = runner.spec_driver_pick([{'role': 'user', 'content': PROMPT}], set(), PROMPT)
    assert pick == {'choice': 'tool', 'tool_name': 'web_fetch', 'p': 0.12}

    # tool pick below MIN_CONF: the router return is empty but top-1 is a tool
    _install_driver_router(monkeypatch, {
        'message': PROMPT, 'choice': 't5', 'tools': [],
        'top': [('web_fetch', 0.12)], 'answer_directly': 0.2, 'ask_user': 0.1,
    }, picks=[])
    pick = runner.spec_driver_pick([], set(), PROMPT)
    assert pick == {'choice': 'tool', 'tool_name': 'web_fetch', 'p': 0.12}

    # answer_directly
    _install_driver_router(monkeypatch, {
        'message': PROMPT, 'choice': 'answer_directly', 'tools': [],
        'top': [('answer_directly', 0.4)], 'answer_directly': 0.4, 'ask_user': 0.1,
    })
    pick = runner.spec_driver_pick([], set(), PROMPT)
    assert pick['choice'] == 'answer_directly' and pick['p'] == 0.4

    # ask_user
    _install_driver_router(monkeypatch, {
        'message': PROMPT, 'choice': 'ask_user', 'tools': [],
        'top': [('ask_user', 0.3)], 'answer_directly': 0.1, 'ask_user': 0.3,
    })
    pick = runner.spec_driver_pick([], set(), PROMPT)
    assert pick['choice'] == 'ask_user'

    # router error / empty decision -> fall back to the normal menu
    _install_driver_router(monkeypatch, {'message': PROMPT, 'error': 'timeout'})
    assert runner.spec_driver_pick([], set(), PROMPT) is None


@pytest.mark.asyncio
async def test_spec_driver_tool_pick_sends_only_that_tool(monkeypatch):
    monkeypatch.setenv('ODYSSEUS_SPEC_MODE', 'driver')
    _install_driver_router(monkeypatch, {
        'message': PROMPT, 'choice': 't1', 'tools': ['web_fetch'],
        'top': [('web_fetch', 0.11)], 'answer_directly': 0.2, 'ask_user': 0.1,
    }, picks=['web_fetch'])
    captured = []
    await _run(monkeypatch, captured)

    assert captured, 'stream_preview made a provider round'
    first = captured[0]
    names = [(s.get('function') or {}).get('name') for s in (first.get('tools') or [])]
    assert names == ['web_fetch']
    assert first.get('tool_choice') == {
        'type': 'function', 'function': {'name': 'web_fetch'}}


@pytest.mark.asyncio
async def test_spec_driver_answer_directly_omits_tools(monkeypatch):
    monkeypatch.setenv('ODYSSEUS_SPEC_MODE', 'driver')
    _install_driver_router(monkeypatch, {
        'message': PROMPT, 'choice': 'answer_directly', 'tools': [],
        'top': [('answer_directly', 0.35)], 'answer_directly': 0.35, 'ask_user': 0.05,
    }, picks=[])
    captured = []
    await _run(monkeypatch, captured)

    assert captured
    assert 'tools' not in captured[0]  # vLLM rejects an empty array


@pytest.mark.asyncio
async def test_spec_driver_router_error_keeps_the_normal_menu(monkeypatch):
    monkeypatch.setenv('ODYSSEUS_SPEC_MODE', 'driver')
    _install_driver_router(monkeypatch, {
        'message': PROMPT, 'choice': None, 'tools': [],
        'top': [], 'answer_directly': 0.0, 'ask_user': 0.0, 'error': 'boom',
    }, picks=[])
    captured = []
    contract, _events = await _run(monkeypatch, captured)

    assert captured
    sent = {(s.get('function') or {}).get('name')
            for s in (captured[0].get('tools') or [])}
    assert sent == set(contract.offered)  # unchanged normal menu
    assert sent


@pytest.mark.asyncio
async def test_spec_driver_pick_uses_the_full_spec_surface(monkeypatch):
    """m3518: Spec picks from the FULL surface (not the compact round menu) and
    the pick is admitted/executable that round (sent as the single tool)."""
    monkeypatch.setenv('ODYSSEUS_SPEC_MODE', 'driver')
    _install_driver_router(monkeypatch, {
        'message': PROMPT, 'choice': 't1', 'tools': ['manage_calendar'],
        'top': [('manage_calendar', 0.9)], 'answer_directly': 0.05, 'ask_user': 0.05,
    }, picks=['manage_calendar'])
    captured = []
    await _run(monkeypatch, captured)

    assert captured
    names = [(s.get('function') or {}).get('name')
             for s in (captured[0].get('tools') or [])]
    assert names == ['manage_calendar']


@pytest.mark.asyncio
async def test_spec_driver_pick_not_on_the_surface_keeps_the_normal_menu(monkeypatch):
    """Only a pick that is truly not executable falls back for the round."""
    monkeypatch.setenv('ODYSSEUS_SPEC_MODE', 'driver')
    _install_driver_router(monkeypatch, {
        'message': PROMPT, 'choice': 't1', 'tools': ['totally_missing_tool'],
        'top': [('totally_missing_tool', 0.9)], 'answer_directly': 0.05, 'ask_user': 0.05,
    }, picks=['totally_missing_tool'])
    captured = []
    contract, _events = await _run(monkeypatch, captured)

    assert captured
    sent = {(s.get('function') or {}).get('name')
            for s in (captured[0].get('tools') or [])}
    assert 'totally_missing_tool' not in sent
    assert sent == set(contract.offered)  # normal menu retained


@pytest.mark.asyncio
async def test_spec_driver_drops_off_pick_native_call(monkeypatch):
    """m3517: execute ONLY the Spec pick; drop any other native call."""
    monkeypatch.setenv('ODYSSEUS_SPEC_MODE', 'driver')
    _install_driver_router(monkeypatch, {
        'message': PROMPT, 'choice': 't1', 'tools': ['web_fetch'],
        'top': [('web_fetch', 0.7)], 'answer_directly': 0.1, 'ask_user': 0.1,
    }, picks=['web_fetch'])
    executed = _install_recorder(monkeypatch)
    scripts = [
        {'tool_calls': [
            {'name': 'web_search', 'arguments': '{"query": "x"}'},
            {'name': 'web_fetch', 'arguments': '{"url": "http://x"}'},
        ]},
        {'content': 'done'},
    ]
    captured = []
    await _run(monkeypatch, captured, scripts=scripts, max_rounds=2)

    assert executed == ['web_fetch'], executed


@pytest.mark.asyncio
async def test_spec_driver_answer_directly_strips_markup_and_executes_nothing(monkeypatch):
    monkeypatch.setenv('ODYSSEUS_SPEC_MODE', 'driver')
    _install_driver_router(monkeypatch, {
        'message': PROMPT, 'choice': 'answer_directly', 'tools': [],
        'top': [('answer_directly', 0.9)], 'answer_directly': 0.9, 'ask_user': 0.05,
    }, picks=[])
    executed = _install_recorder(monkeypatch)
    scripts = [{
        'content': 'Hello <tool_call>{"name": "web_search", "arguments": {}}</tool_call> world',
    }]
    captured = []
    _contract_, events = await _run(monkeypatch, captured, scripts=scripts)

    assert executed == []
    finals = [e.get('content', '') for e in events if e.get('type') == 'final_response']
    assert finals, events
    assert 'Hello' in finals[-1] and '<tool_call' not in finals[-1]


def test_spec_driver_wired_in_stream_preview():
    src = Path(runner.__file__).read_text()
    assert '[spec-driver] round=' in src
    assert 'spec_driver_pick(' in src
    assert '[spec-driver] dropped off-pick call name=' in src


def test_spec_contract_offered_extends_the_model_runtime(monkeypatch):
    """A Spec pick outside PREVIEW_TOOLS must still be runnable: on a
    spec-router turn the contract's offered set is part of the runtime."""
    import types
    from src import turn_contract as tc
    from src.clean_agent_preview import PREVIEW_TOOLS, evaluate_preview_call

    assert 'get_weather' not in PREVIEW_TOOLS

    class _C:
        routing_experiment = 'spec-router'
        offered = frozenset({'get_weather'})

    monkeypatch.setattr(tc, 'active_turn_contract', lambda: _C())
    allowed = evaluate_preview_call(
        'get_weather', {'location': 'Tokyo'}, 'weather',
        turn_authorized_families=frozenset(),
    )
    assert allowed.allowed, allowed

    class _Plain:
        routing_experiment = 'baseline'
        offered = frozenset({'get_weather'})

    monkeypatch.setattr(tc, 'active_turn_contract', lambda: _Plain())
    denied = evaluate_preview_call(
        'get_weather', {'location': 'Tokyo'}, 'weather',
        turn_authorized_families=frozenset(),
    )
    assert not denied.allowed
    assert denied.reason == 'tool_not_in_model_runtime'
