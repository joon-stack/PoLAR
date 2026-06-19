__version__ = "0.1.0"
__project__ = "polar-vla"
__author__ = "Youngjoon Jeong, Jihwan Yu, Minsoo Jo, Junha Chun, Taesup Kim"
__license__ = "Apache License 2.0"
__email__ = ""


_MODEL_EXPORTS = {
    "available_model_names",
    "available_models",
    "get_model_description",
    "load",
}


def __getattr__(name):
    if name in _MODEL_EXPORTS:
        from . import models

        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_MODEL_EXPORTS)
