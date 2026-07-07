"""Cognitive modules -- the replaceable "organs" of the organism.

Every module is a self-contained plugin that only ever communicates through the
event bus.  New modules can be added without touching the kernel; see
:mod:`modules.registry` for how they are discovered and built.
"""
