"""Async utilities for Streamlit WebUI.

This module provides safe async execution patterns for Streamlit,
which runs in a synchronous context but needs to call async code.

The key issue is that creating and closing event loops repeatedly
causes problems with libraries like LightRAG that maintain async state
(connections, queues, etc.) across calls.
"""

import asyncio
import threading
from typing import TypeVar, Coroutine, Any

T = TypeVar('T')

# Global event loop management
_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def get_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create a shared event loop for async operations.
    
    This ensures we reuse the same event loop across all async calls,
    preventing 'Event loop is closed' errors from libraries that
    maintain async state.
    """
    global _loop
    
    with _loop_lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
        return _loop


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Safely run an async coroutine from synchronous code.
    
    This function:
    1. Uses a shared event loop instead of creating new ones
    2. Does NOT close the loop after execution
    3. Handles nested event loop scenarios
    
    Args:
        coro: The coroutine to execute
        
    Returns:
        The result of the coroutine
        
    Raises:
        Any exception raised by the coroutine
    """
    loop = get_event_loop()
    
    # Check if we're already in an async context
    try:
        running_loop = asyncio.get_running_loop()
        if running_loop is loop:
            # We're inside the same loop - create a task
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No running loop - this is the normal case
        pass
    
    # Set the loop and run
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def cleanup_event_loop() -> None:
    """Clean up the shared event loop.
    
    Call this when the application is shutting down.
    """
    global _loop
    
    with _loop_lock:
        if _loop is not None and not _loop.is_closed():
            try:
                # Cancel all pending tasks
                pending = asyncio.all_tasks(_loop)
                for task in pending:
                    task.cancel()
                
                # Run until all tasks are cancelled
                if pending:
                    _loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                
                _loop.close()
            except Exception:
                pass  # Ignore cleanup errors
            finally:
                _loop = None
