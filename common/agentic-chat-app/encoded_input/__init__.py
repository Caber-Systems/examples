"""
Encoded Input Component -- XOR-encodes a default question server-side so that
PyCaber's n-mer extraction cannot match the plaintext in the Streamlit WS frame.
The browser decodes it client-side before displaying to the user.
"""

import os
import secrets
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))
_component_func = components.declare_component("encoded_input", path=_COMPONENT_DIR)


def xor_encode(plaintext: str, key: bytes) -> list[int]:
    """XOR each UTF-8 byte of plaintext with the key (cycling)."""
    raw = plaintext.encode("utf-8")
    return [b ^ key[i % len(key)] for i, b in enumerate(raw)]


def encoded_text_area(
    label: str,
    default_value: str = "",
    height: int = 100,
    key: str | None = None,
) -> str | None:
    """
    Render a text area whose default value is XOR-encoded over the wire.

    The server sends (encoded_bytes, xor_key) to the frontend JS.  The JS
    decodes and displays plaintext.  PyCaber sees only the XOR-scrambled
    bytes in the Streamlit WS frame, so no n-mer match occurs between the
    server-rendered default and the user's subsequent form submission.

    Returns the user-entered plaintext (or None if no interaction yet).
    """
    xor_key = secrets.token_bytes(32)

    encoded = xor_encode(default_value, xor_key) if default_value else []

    result = _component_func(
        label=label,
        encoded=encoded,
        xor_key=list(xor_key),
        height=height,
        key=key,
        default=None,
    )
    return result
