"""Biologically inspired memory subsystem.

Working memory (small, volatile), episodic memory (experiences with decay) and
semantic memory (consolidated concepts) sit behind a single
:class:`memory.manager.MemoryManager` facade that listens to the event bus and
forms memories automatically -- no module ever touches memory directly.
"""
