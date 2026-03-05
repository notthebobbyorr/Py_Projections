"""Projection package (v1)."""


def run_projection_pipeline(*args, **kwargs):
    from .run import run_projection_pipeline as _run_projection_pipeline

    return _run_projection_pipeline(*args, **kwargs)


__all__ = ["run_projection_pipeline"]
