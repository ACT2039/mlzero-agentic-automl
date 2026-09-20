
import pytest

from mlzero.schemas.perception import (
    DataQualityReport,
    LibrarySelection,
    PerceptualContext,
    TaskContext,
)
from mlzero.tools.adapters import AdapterRegistry
from mlzero.tools.adapters.general import GeneralMLAdapter
from mlzero.tools.adapters.multimodal import MultiModalAdapter
from mlzero.tools.adapters.retrieval import RetrievalAdapter
from mlzero.tools.adapters.tabular import TabularAdapter
from mlzero.tools.adapters.timeseries import TimeSeriesAdapter


def create_mock_context(library_name: str) -> PerceptualContext:
    return PerceptualContext(
        files=[],
        task=TaskContext(),
        library=LibrarySelection(selected_library=library_name, explanation="mock"),
        data_quality=DataQualityReport()
    )

def test_adapter_registry_discovery():
    assert len(AdapterRegistry._adapters) == 5

def test_adapter_selection_tabular():
    ctx = create_mock_context("autogluon.tabular")
    adapter = AdapterRegistry.get_adapter(ctx)
    assert isinstance(adapter, TabularAdapter)
    
def test_adapter_selection_timeseries():
    ctx = create_mock_context("autogluon.timeseries")
    adapter = AdapterRegistry.get_adapter(ctx)
    assert isinstance(adapter, TimeSeriesAdapter)

def test_adapter_selection_multimodal():
    ctx = create_mock_context("autogluon.multimodal")
    adapter = AdapterRegistry.get_adapter(ctx)
    assert isinstance(adapter, MultiModalAdapter)
    
def test_adapter_selection_retrieval():
    ctx = create_mock_context("FlagEmbedding")
    adapter = AdapterRegistry.get_adapter(ctx)
    assert isinstance(adapter, RetrievalAdapter)
    
def test_adapter_selection_general_ml():
    ctx = create_mock_context("machine learning")
    adapter = AdapterRegistry.get_adapter(ctx)
    assert isinstance(adapter, GeneralMLAdapter)
    
    ctx = create_mock_context("general_ml")
    adapter = AdapterRegistry.get_adapter(ctx)
    assert isinstance(adapter, GeneralMLAdapter)

def test_tabular_adapter_invalid_input(tmp_path):
    ctx = create_mock_context("autogluon.tabular")
    adapter = TabularAdapter()
    
    # Missing train file
    with pytest.raises(ValueError, match="No valid tabular training data found"):
        adapter.prepare_data(tmp_path, tmp_path / "work", ctx)
        
def test_expected_output_contracts():
    adapters = [TabularAdapter(), MultiModalAdapter(), TimeSeriesAdapter()]
    for adapter in adapters:
        outputs = adapter.expected_outputs()
        assert "out/models" in outputs
        assert "out/predictions.csv" in outputs
        
    retrieval_adapter = RetrievalAdapter()
    assert "out/retrieval_results.json" in retrieval_adapter.expected_outputs()
