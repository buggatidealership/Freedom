"""Exceptions that mean "a prerequisite another command creates is missing".

The CLI turns these (and FileNotFoundError) into exit code 2 with a message naming the command
to run first; anything else — KeyError, IndexError, ... — is a programming or data-shape error
and propagates with a traceback. This module has no heavy imports so `freedom.cli` can catch
the types without importing the modules that raise them.
"""

from __future__ import annotations


class EventNotFound(LookupError):
    """An event id is neither in data/events.parquet nor in the upcoming calendar."""


__all__ = ["EventNotFound"]
