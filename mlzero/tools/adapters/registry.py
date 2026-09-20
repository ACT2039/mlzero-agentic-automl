from typing import ClassVar

from mlzero.schemas.perception import PerceptualContext
from mlzero.tools.adapters.base import MLLibraryAdapter
from mlzero.tools.adapters.general import GeneralMLAdapter
from mlzero.tools.adapters.multimodal import MultiModalAdapter
from mlzero.tools.adapters.retrieval import RetrievalAdapter
from mlzero.tools.adapters.tabular import TabularAdapter
from mlzero.tools.adapters.timeseries import TimeSeriesAdapter


class AdapterRegistry:
    """Registry to discover and instantiate execution adapters."""

    _adapters: ClassVar[list[type[MLLibraryAdapter]]] = [
        TabularAdapter,
        MultiModalAdapter,
        TimeSeriesAdapter,
        RetrievalAdapter,
        GeneralMLAdapter
    ]

    @classmethod
    def get_adapter(cls, perceptual_context: PerceptualContext) -> MLLibraryAdapter | None:
        """Return the first adapter that supports the context."""
        for adapter_class in cls._adapters:
            adapter = adapter_class()
            if adapter.supports(perceptual_context):
                return adapter
        return None
