"""FR3 SigLIP + flow-matching inference integration (100k weights)."""

from .configuration_flowmatch import FlowMatchConfig as FlowMatchConfig

__all__ = ["FlowMatchConfig", "FlowMatchPolicy", "register", "make_flowmatch_pre_post_processors"]


def register():
    """Importing this package registers flowmatch with the existing LeRobot registry."""
    return FlowMatchConfig


def __getattr__(name):
    if name == "FlowMatchPolicy":
        from .modeling_flowmatch import FlowMatchPolicy

        return FlowMatchPolicy
    if name == "make_flowmatch_pre_post_processors":
        from .processor_flowmatch import make_flowmatch_pre_post_processors

        return make_flowmatch_pre_post_processors
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
