#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Types shared across Pipecat, independent of any provider SDK.

The ``NOT_GIVEN`` sentinel lives here so that settings, contexts and anything
else needing "this value was not provided" share one meaning. Provider SDKs
ship their own equivalents; those belong to the SDK and are translated at the
adapter boundary rather than used inside Pipecat.
"""

from typing import Literal, TypeGuard, TypeVar


# This fork defines the sentinel in pipecat.services.settings; reuse that one instance so
# settings objects and the backported OpenAI Live service agree on what "not given" is.
from pipecat.services.settings import NOT_GIVEN as NOT_GIVEN  # noqa: E402
from pipecat.services.settings import _NotGiven as NotGiven  # noqa: E402

_T = TypeVar("_T")


def is_given(value: _T | NotGiven) -> TypeGuard[_T]:
    """Check whether a value was explicitly provided.

    Also acts as a type guard: inside a true branch, the value is narrowed to
    exclude :class:`NotGiven` (e.g. ``str | None | NotGiven`` becomes
    ``str | None``)::

        if is_given(delta.voice):
            # caller wants to change the voice
            ...

    Args:
        value: The value to check.

    Returns:
        ``True`` if *value* is anything other than ``NOT_GIVEN``.
    """
    return not isinstance(value, NotGiven)


def assert_given(value: _T | NotGiven) -> _T:
    """Extract a value that must have been provided.

    Intended for reads where ``NOT_GIVEN`` should never appear, such as a
    store-mode settings object (see :mod:`pipecat.services.settings`). Narrows
    away :class:`NotGiven` at the type level and raises at runtime if that
    invariant is violated::

        resolved_model = assert_given(self._settings.model)  # narrowed str | None

    Args:
        value: The value to extract.

    Returns:
        The value, narrowed to exclude :class:`NotGiven`.

    Raises:
        RuntimeError: If *value* is ``NOT_GIVEN``.
    """
    if not is_given(value):
        raise RuntimeError("Expected a value, got NOT_GIVEN")
    return value


def require_given(value: _T | None | NotGiven, what: str) -> _T:
    """Extract a value that must have been provided and must not be empty.

    Like :func:`assert_given`, but also rejects ``None`` and the empty string,
    which is what an unset or blank environment variable produces. Use it for
    settings a service cannot work without, such as a local model or voice
    that no provider will validate on its behalf::

        voice = require_given(self._settings.voice, "Piper TTS voice")  # narrowed str

    Args:
        value: The value to extract.
        what: How to name the value in the error, e.g. ``"Piper TTS voice"``.

    Returns:
        The value, narrowed to exclude :class:`NotGiven` and ``None``.

    Raises:
        RuntimeError: If *value* is ``NOT_GIVEN``.
        ValueError: If *value* is ``None`` or an empty string.
    """
    value = assert_given(value)
    if value is None or value == "":
        raise ValueError(f"{what} must be specified")
    return value
