from contextlib import contextmanager

from rag import config

ENABLED = bool(config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY)

if ENABLED:
    from langfuse import get_client

    _client = get_client()
else:
    _client = None


@contextmanager
def observation(name: str, as_type: str = "span", **kwargs):
    """A Langfuse observation, or a no-op if LANGFUSE_PUBLIC_KEY/SECRET_KEY aren't set.

    The first observation opened in a call stack becomes the trace root; anything opened while
    it's active (dense/hybrid retrieval inside a query, generation inside that, etc.) nests under
    it automatically via OpenTelemetry context propagation, no trace/span ids to pass by hand.
    Yields None when disabled, so callers must guard updates with `if obs:`.
    """
    if not ENABLED:
        yield None
        return
    with _client.start_as_current_observation(name=name, as_type=as_type, **kwargs) as obs:
        yield obs


def score_current_trace(name: str, value: float | str, **kwargs):
    if not ENABLED:
        return
    _client.score_current_trace(name=name, value=value, **kwargs)


def flush():
    """Force-send any buffered traces. Call at the end of short-lived CLI processes, the API
    flushes on shutdown instead."""
    if ENABLED:
        _client.flush()
