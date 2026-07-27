#!/usr/bin/env python3
"""Bounded physical-to-logical PROMPTBOOT_EVENT topology parsing."""

from __future__ import annotations

from typing import NamedTuple

PREFIX = b"PROMPTBOOT_EVENT v=1 "
TOGGLE_V1 = "toggle-v1"
STRICT_FINAL = "strict_final"
POLLING_PREFIX = "polling_prefix"
MODES = (STRICT_FINAL, POLLING_PREFIX)
STATUS_ON = b"events: on\r\n"
STATUS_OFF = b"events: off\r\n"


class TopologyResult(NamedTuple):
    logical_records: tuple[bytes, ...]
    physical_records: tuple[bytes, ...]
    pending_record: bytes | None = None

    def counts(self) -> dict[str, int]:
        return {
            "logical_records": len(self.logical_records),
            "physical_records": len(self.physical_records),
        }


def _items(
    payload: bytes, *, allow_incomplete_tail: bool = False
) -> list[tuple[str, bytes, int, int]]:
    items = []
    start = 0
    for line in payload.splitlines(keepends=True):
        end = start + len(line)
        at = line.find(PREFIX)
        if at >= 0:
            record = line[at:]
            if not record.endswith(b"\r\n"):
                if allow_incomplete_tail and end == len(payload):
                    start = end
                    continue
                raise ValueError("incomplete physical event record")
            items.append(("record", record, start + at, end))
        elif line == STATUS_ON:
            items.append(("on", line, start, end))
        elif line == STATUS_OFF:
            items.append(("off", line, start, end))
        start = end
    return items


def event_lines(payload: bytes) -> list[bytes]:
    """Return every complete raw event copy without assigning logical meaning."""
    return [value for kind, value, _start, _end in _items(payload) if kind == "record"]


def parse_records(payload: bytes, topology: str, mode: str) -> TopologyResult:
    """Parse one bounded raw snapshot under an explicit final/polling mode."""
    if mode not in MODES:
        raise ValueError(f"unknown event parse mode {mode!r}")
    if topology != TOGGLE_V1:
        raise ValueError(f"unknown event topology {topology!r}")

    items = _items(payload, allow_incomplete_tail=mode == POLLING_PREFIX)
    physical = tuple(value for kind, value, _start, _end in items if kind == "record")
    logical = []
    display_enabled = False
    at = 0
    while at < len(items):
        kind, value, _start, end = items[at]
        if kind == "on":
            if display_enabled:
                raise ValueError("duplicate events: on boundary")
            display_enabled = True
            at += 1
            continue
        if kind == "off":
            if not display_enabled:
                raise ValueError("events: off boundary while already off")
            display_enabled = False
            at += 1
            continue
        if display_enabled:
            if at + 1 == len(items):
                if mode == POLLING_PREFIX:
                    return TopologyResult(tuple(logical), physical, pending_record=value)
                raise ValueError("enabled event has one unauthorized dangling physical copy")
            next_kind, next_value, next_start, _next_end = items[at + 1]
            if next_kind != "record":
                raise ValueError("enabled event copies are not adjacent")
            if end != next_start:
                raise ValueError("enabled event copies have intervening bytes")
            if next_value != value:
                raise ValueError("enabled event copies differ")
            logical.append(value)
            at += 2
            continue
        if at + 1 < len(items):
            next_kind, next_value, _next_start, _next_end = items[at + 1]
            if next_kind == "record" and next_value == value:
                raise ValueError("disabled event has a physical pair")
        logical.append(value)
        at += 1
    return TopologyResult(tuple(logical), physical)
