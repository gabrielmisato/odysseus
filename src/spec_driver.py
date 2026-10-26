"""Ajax Spec turn: the Spec router drives every step; the chat model never chooses a tool.

Each round the router (spec_router.next_action) picks one action from the full tool surface:
  tool            -> the chat model gets only that tool (named tool_choice) and writes its arguments;
                     the executor runs it; the result goes back to the router for the next pick.
  answer_directly -> the chat model writes the answer with no tools (tools key omitted).
  ask_user        -> the chat model writes the question through the ask_user tool.
Router unavailable -> the chat model answers without tools. Execution keeps the normal request
authority and tool policy (execute_tool_block); nothing else is layered on top.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid

import httpx

from src import spec_router
from src.agent_runtime.authority import MISSING_AUTHORITY, with_request_authority
from src.clean_agent_preview import (
    event, preview_model_response, preview_tool_result_text,
)
from src.tool_capabilities import ToolRunSecurityContext
from src.tool_execution import execute_tool_block
from src.tool_schemas import FUNCTION_TOOL_SCHEMAS, function_call_to_tool_block

logger = logging.getLogger(__name__)


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(str(p.get('text') or '') for p in content
                         if isinstance(p, dict) and p.get('type') == 'text')
    return ''


def _name(schema):
    return (schema.get('function') or {}).get('name') or ''


# JSON-schema keywords the provider's guided decoding (named tool_choice) cannot compile.
_GRAMMAR_UNSUPPORTED = frozenset({
    'uniqueItems', 'contains', 'minContains', 'maxContains', 'patternProperties', 'propertyNames',
    'dependentRequired', 'dependentSchemas', 'if', 'then', 'else',
    'unevaluatedProperties', 'unevaluatedItems',
})


def grammar_safe(schema):
    """Copy of a schema without keywords guided decoding rejects (validation still runs at execution)."""
    if isinstance(schema, dict):
        return {k: grammar_safe(v) for k, v in schema.items() if k not in _GRAMMAR_UNSUPPORTED}
    if isinstance(schema, list):
        return [grammar_safe(v) for v in schema]
    return schema


def spec_surface(disabled_tools):
    disabled = set(disabled_tools or ())
    return [s for s in FUNCTION_TOOL_SCHEMAS if _name(s) and _name(s) not in disabled]


async def _model_round(client, endpoint_url, headers, request, recovery):
    """Run one chat-model round; return (content, tool_calls, usage)."""
    content, calls, usage, raw = '', {}, {}, []
    async with preview_model_response(client, endpoint_url, headers, request, recovery) as response:
        async for line in response.aiter_lines():
            if len(raw) < 4 and line.strip():
                raw.append(line[:400])
            if line.strip() and not line.startswith('data: '):
                logger.warning('[spec-driver] unexpected stream line: %s', line[:500])
            if not line.startswith('data: ') or line.strip() == 'data: [DONE]':
                continue
            try:
                chunk = json.loads(line[6:])
            except ValueError:
                continue
            if chunk.get('error') or chunk.get('object') == 'error':
                raise RuntimeError(f"model stream error: {str(chunk.get('error') or chunk)[:500]}")
            usage = chunk.get('usage') or usage
            for choice in chunk.get('choices') or []:
                delta = choice.get('delta') or {}
                text = delta.get('content') or ''
                content += text
                for frag in delta.get('tool_calls') or []:
                    call = calls.setdefault(frag.get('index', 0), {'id': '', 'name': '', 'arguments': ''})
                    call['id'] = frag.get('id') or call['id']
                    fn = frag.get('function') or {}
                    call['name'] += fn.get('name') or ''
                    call['arguments'] += fn.get('arguments') or ''
    if not calls:
        logger.warning('[spec-driver] no tool call; status=%s max_tokens=%s msgs=%s first lines=%s',
                       getattr(response, 'status_code', None), request.get('max_tokens'),
                       [(m.get('role'), type(m.get('content')).__name__,
                         [p.get('type') for p in m['content']] if isinstance(m.get('content'), list) else len(str(m.get('content') or '')))
                        for m in request.get('messages', [])], raw[-2:])
    return content, [calls[i] for i in sorted(calls)], usage


@with_request_authority
async def stream_spec_turn(*, endpoint_url, model, messages, headers, session_id, owner,
                           disabled_tools, tool_policy, workspace=None, active_document=None,
                           client_runtime_context=None, external_untrusted_context_seen=False,
                           max_tokens=768, max_rounds=8, temperature=0.0,
                           request_authority=MISSING_AUTHORITY, **ignored):
    from src.turn_contract import _ACTIVE_CONTRACT
    _ACTIVE_CONTRACT.set(None)  # Spec, not the regex turn contract, decides the tools.
    started = time.monotonic()
    surface = spec_surface(disabled_tools)
    by_name = {_name(s): s for s in surface}
    user_text = next((_text(m.get('content')) for m in reversed(messages) if m.get('role') == 'user'), '')
    history = list(messages)
    security = ToolRunSecurityContext(
        external_untrusted_context_seen=bool(external_untrusted_context_seen), unattended_tools=frozenset())
    security.observe_messages(messages)
    picks, executions, done_calls = [], [], set()
    latest, first_token, usage_in, usage_out, rounds = '', None, 0, 0, 0
    recovery = {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        for round_number in range(1, max_rounds + 1):
            rounds = round_number
            action = await asyncio.to_thread(
                spec_router.next_action, user_text, latest, surface, messages[:-1], disabled_tools)
            if action is None:
                action = {'kind': 'answer_directly', 'name': 'router_unavailable', 'p': None}
            if round_number == max_rounds and action['kind'] == 'tool':
                action = {'kind': 'answer_directly', 'name': 'round_limit', 'p': None}
            picks.append({'round': round_number, **action})
            logger.info('[spec-driver] round=%s pick=%s p=%s', round_number, action['name'], action.get('p'))
            request = {'model': model, 'messages': history, 'max_tokens': max_tokens,
                       'temperature': temperature, 'stream': True,
                       'stream_options': {'include_usage': True},
                       'chat_template_kwargs': {'enable_thinking': False}}

            if action['kind'] == 'answer_directly':
                answer_parts = []
                async with preview_model_response(client, endpoint_url, headers, request, recovery) as response:
                    async for line in response.aiter_lines():
                        if not line.startswith('data: ') or line.strip() == 'data: [DONE]':
                            continue
                        try:
                            chunk = json.loads(line[6:])
                        except ValueError:
                            continue
                        if chunk.get('error') or chunk.get('object') == 'error':
                            raise RuntimeError(f"model stream error: {str(chunk.get('error') or chunk)[:500]}")
                        u = chunk.get('usage') or {}
                        usage_in += u.get('prompt_tokens') or 0
                        usage_out += u.get('completion_tokens') or 0
                        for choice in chunk.get('choices') or []:
                            text = (choice.get('delta') or {}).get('content') or ''
                            if text:
                                first_token = first_token or time.monotonic()
                                answer_parts.append(text)
                                yield event({'delta': text})
                answer = ''.join(answer_parts).strip()
                if not answer:
                    yield event({'delta': 'The model returned no answer.'})
                history.append({'role': 'assistant', 'content': answer})
                break

            name = 'ask_user' if action['kind'] == 'ask_user' else action['name']
            # ODYSSEUS_SPEC_DRIVER_TOPK=k>1: offer Spec's top-k tools and let the chat model choose among them.
            topk = int(os.getenv('ODYSSEUS_SPEC_DRIVER_TOPK', '1') or 1)
            options = [name]
            if action['kind'] == 'tool' and topk > 1:
                options = [n for n, _ in action.get('top') or []][:topk] or [name]
            schemas = [by_name.get(n) or next((s for s in FUNCTION_TOOL_SCHEMAS if _name(s) == n), None)
                       for n in options]
            options = [n for n, sc in zip(options, schemas) if sc is not None]
            schemas = [sc for sc in schemas if sc is not None]
            if not schemas:
                logger.info('[spec-driver] pick %s has no schema; answering instead', name)
                latest = f'Tool {name} is not available.'
                continue
            request['tools'] = [grammar_safe(sc) for sc in schemas]
            request['tool_choice'] = ({'type': 'function', 'function': {'name': options[0]}}
                                      if len(options) == 1 else 'required')
            if len(options) > 1:
                logger.info('[spec-driver] round=%s options=%s', round_number, options)
            try:
                content, calls, usage = await _model_round(client, endpoint_url, headers, request, recovery)
            except (RuntimeError, httpx.HTTPError) as exc:
                logger.warning('[spec-driver] round=%s %s call failed: %s', round_number, name, exc)
                latest = f'The {name} call could not be produced: {exc}'
                continue
            first_token = first_token or time.monotonic()
            usage_in += usage.get('prompt_tokens') or 0
            usage_out += usage.get('completion_tokens') or 0
            call = next((c for c in calls if c['name'] in options), None)
            if call is not None and len(options) > 1:
                name = call['name']
                logger.info('[spec-driver] round=%s model chose=%s rank=%s of %s',
                            round_number, name, options.index(name) + 1, options)
                picks[-1]['chosen'] = name
            if call is None:
                latest = f'The {name} call was not produced.'
                logger.info('[spec-driver] round=%s no %s call from the model', round_number, name)
                continue
            arguments = call['arguments'] or '{}'
            signature = (name, arguments)
            call_id = call['id'] or f'call_{uuid.uuid4().hex[:12]}'
            history.append({'role': 'assistant', 'content': content or '',
                            'tool_calls': [{'id': call_id, 'type': 'function',
                                            'function': {'name': name, 'arguments': arguments}}]})
            if signature in done_calls:
                output = 'This exact call already succeeded; its result is above. Do not repeat it.'
                history.append({'role': 'tool', 'tool_call_id': call_id, 'content': output})
                latest = output
                continue
            try:
                args = json.loads(arguments)
            except ValueError:
                args = {}
            block = function_call_to_tool_block(name, arguments)
            yield event({'type': 'tool_start', 'tool': name, 'command': arguments,
                         'full_command': arguments, 'round': round_number})
            tool_started = time.monotonic()
            if block is None:
                desc, result = f'{name}: invalid arguments', {
                    'error': 'Tool arguments could not be converted for execution.', 'exit_code': 1}
            else:
                desc, result = await execute_tool_block(
                    block, session_id=session_id, owner=owner, disabled_tools=disabled_tools,
                    tool_policy=tool_policy, security_context=security,
                    active_document_id=getattr(active_document, 'id', None), workspace=workspace,
                    client_runtime_context=client_runtime_context, request_authority=request_authority)
            failed = bool(result.get('error')) or result.get('exit_code') not in (None, 0)
            output = preview_tool_result_text(result, name, args)
            tool_event = {'type': 'tool_output', 'tool': name, 'command': arguments, 'output': output,
                          'exit_code': result.get('exit_code', 1 if failed else 0), 'error': failed,
                          'execution_attempted': block is not None, 'blocked': bool(result.get('blocked')),
                          'desc': desc, 'round': round_number,
                          'seconds': round(time.monotonic() - tool_started, 3)}
            if result.get('image_url'):
                yield event({'type': 'generated_image', 'url': result['image_url'],
                             **{k: result[k] for k in ('image_url', 'image_id', 'image_prompt') if k in result}})
            if result.get('ui_event'):
                yield event({'type': 'ui_control', 'data': result})
            executions.append(tool_event)
            yield event(tool_event)
            history.append({'role': 'tool', 'tool_call_id': call_id, 'content': output})
            if not failed:
                done_calls.add(signature)
            latest = output
            ask = result.get('ask_user')
            if isinstance(ask, dict) and str(ask.get('question') or '').strip():
                question = str(ask['question']).strip()
                history.append({'role': 'assistant', 'content': question})
                yield event({'type': 'final_response', 'content': question})
                yield event({'type': 'ask_user', 'data': ask})
                break
    elapsed = time.monotonic() - started
    yield event({'type': 'metrics', 'data': {
        'model': model, 'input_tokens': usage_in, 'output_tokens': usage_out,
        'total_tokens': usage_in + usage_out, 'response_time': round(elapsed, 3),
        'time_to_first_token': round(first_token - started, 3) if first_token else None,
        'tokens_per_second': round(usage_out / elapsed, 2) if elapsed > 0 else 0,
        'agent_rounds': rounds, 'tool_calls': len(executions), 'tool_events': executions,
        'spec_driver_picks': picks, 'tool_schema_count': len(surface),
        'schema_mode': 'spec_driver', 'temperature': temperature, 'max_output_tokens': max_tokens,
    }})
    yield 'data: [DONE]\n\n'
