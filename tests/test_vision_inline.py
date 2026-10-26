"""Inline-image vision path: direct image_url parts, downscale, probe, trimming.

Covers the change that sends attached images straight to the chat model as an
OpenAI image_url content part (instead of a separate vision-describe pre-step)
when the endpoint/model accepts images.
"""
from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from src import chat_helpers as ch
from src import document_processor as dp
from src.context_compactor import keep_recent_image_turns


def _png_bytes(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (200, 40, 40)).save(buf, format="PNG")
    return buf.getvalue()


# --- vision detection -------------------------------------------------------

def test_vision_model_env_override(monkeypatch):
    monkeypatch.delenv("ODYSSEUS_VISION_MODELS", raising=False)
    assert ch.is_vision_model("ajax-ablated500-base") is False
    monkeypatch.setenv("ODYSSEUS_VISION_MODELS", "ajax, some-other")
    assert ch.is_vision_model("ajax-ablated500-base") is True
    assert ch.is_vision_model("SOME-OTHER-9B") is True
    assert ch.is_vision_model("plain-text-model") is False


def test_probe_endpoint_supports_vision_cached(monkeypatch):
    ch._vision_probe_cache.clear()
    calls = []

    class _Resp:
        def __init__(self, status, text=""):
            self.status_code = status
            self.text = text

    def _fake_post(url, json=None, timeout=None):
        calls.append(url)
        return _Resp(200, "")

    monkeypatch.setattr(ch.httpx, "post", _fake_post)
    assert ch.probe_endpoint_supports_vision("http://127.0.0.1:18607/v1", "ajax-ablated500-base") is True
    # Second call is served from the cache (no extra POST).
    assert ch.probe_endpoint_supports_vision("http://127.0.0.1:18607/v1", "ajax-ablated500-base") is True
    assert len(calls) == 1
    assert calls[0].endswith("/v1/chat/completions")


def test_probe_endpoint_rejects_image(monkeypatch):
    ch._vision_probe_cache.clear()

    class _Resp:
        status_code = 400
        text = '{"error":"this model does not support image content"}'

    monkeypatch.setattr(ch.httpx, "post", lambda *a, **k: _Resp())
    assert ch.probe_endpoint_supports_vision("http://127.0.0.1:18607/v1", "text-only-9b") is False


def test_probe_never_probes_remote_hosts(monkeypatch):
    ch._vision_probe_cache.clear()
    calls = []
    monkeypatch.setattr(ch.httpx, "post", lambda *a, **k: calls.append(a) or None)
    assert ch.probe_endpoint_supports_vision("https://api.openai.com/v1", "gpt-x") is None
    assert calls == []


def test_model_supports_vision_uses_probe_for_local(monkeypatch):
    monkeypatch.delenv("ODYSSEUS_VISION_MODELS", raising=False)
    monkeypatch.setattr(ch, "lmstudio_supports_vision", lambda *a, **k: None)
    monkeypatch.setattr(ch, "probe_endpoint_supports_vision", lambda url, model: True)
    assert ch.model_supports_vision("ajax-ablated500-base", "http://127.0.0.1:18607/v1") is True


# --- downscale --------------------------------------------------------------

def test_encode_image_data_url_downscales_to_jpeg(tmp_path):
    path = tmp_path / "big.png"
    path.write_bytes(_png_bytes(2400, 600))
    data_url, fmt = dp.encode_image_data_url(str(path), "png")
    assert fmt == "jpeg"
    assert data_url.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(data_url.split(",", 1)[1])
    with Image.open(io.BytesIO(raw)) as img:
        assert max(img.size) <= 1280
        # aspect ratio preserved
        assert round(img.size[0] / img.size[1], 2) == 4.0


def test_encode_image_data_url_small_image_ok(tmp_path):
    path = tmp_path / "small.png"
    path.write_bytes(_png_bytes(64, 48))
    data_url, fmt = dp.encode_image_data_url(str(path), "png")
    assert fmt == "jpeg"
    raw = base64.b64decode(data_url.split(",", 1)[1])
    with Image.open(io.BytesIO(raw)) as img:
        assert img.size == (64, 48)


# --- keep last N image turns ------------------------------------------------

def _user_with_image(caption: str, text: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
        "_caption": caption,
    }


def test_keep_recent_image_turns_labels_old_images():
    msgs = [
        {"role": "user", "content": "hello"},
        _user_with_image("a.png", "[Image attached: a.png]\nlook at a"),
        {"role": "assistant", "content": "sure"},
        _user_with_image("b.png", "[Image attached: b.png]\nlook at b"),
        _user_with_image("c.png", "[Image attached: c.png]\nlook at c"),
    ]
    out = keep_recent_image_turns(msgs, keep=2)

    def _img_count(m):
        c = m.get("content")
        return sum(1 for b in c if isinstance(b, dict) and b.get("type") == "image_url") if isinstance(c, list) else 0

    # first (oldest) image turn lost its image part, gained the label
    assert _img_count(out[1]) == 0
    assert "[Image: a.png]" in "".join(
        b.get("text", "") for b in out[1]["content"] if isinstance(b, dict)
    )
    # two most recent image turns keep their image parts
    assert _img_count(out[3]) == 1
    assert _img_count(out[4]) == 1
    # non-image message untouched (identity preserved for strings)
    assert out[0]["content"] == "hello"


def test_keep_recent_image_turns_noop_when_few_images():
    msgs = [_user_with_image("a.png", "[Image attached: a.png]")]
    out = keep_recent_image_turns(msgs, keep=2)
    assert out == msgs
