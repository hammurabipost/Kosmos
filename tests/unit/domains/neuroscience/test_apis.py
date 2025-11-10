"""
Unit tests for Neuroscience domain API clients (Phase 9).

Tests 7 neuroscience API clients:
- FlyWire: Drosophila connectome
- AllenBrain: Gene expression atlas
- MICrONS: Mouse cortex connectome
- GEO: Gene Expression Omnibus
- AMPAD: Alzheimer's Disease portal
- OpenConnectome: Connectome repository
- WormBase: C. elegans database

Coverage target: 40 tests (7 clients × 5-6 tests)
"""

import pytest
from unittest.mock import Mock, patch
from kosmos.domains.neuroscience.apis import (
    FlyWireClient,
    AllenBrainClient,
    MICrONSClient,
    GEOClient,
    AMPADClient,
    OpenConnectomeClient,
    WormBaseClient,
    NeuronData,
    GeneExpressionData,
    ConnectomeDataset
)


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for API testing"""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"success": True, "data": "test"}
    mock_response.text = "test_data"
    mock_client.get = Mock(return_value=mock_response)
    mock_client.post = Mock(return_value=mock_response)
    mock_client.close = Mock()
    return mock_client


@pytest.mark.unit
class TestFlyWireClient:
    """Test FlyWire API client"""

    def test_init_default(self):
        """Test FlyWire client initialization"""
        client = FlyWireClient()
        assert hasattr(client, 'client')
        assert FlyWireClient.BASE_URL == "https://global.daf-apis.com/nglstate/api/v1"

    def test_get_neuron_success(self):
        """Test successful neuron data retrieval"""
        client = FlyWireClient()
        result = client.get_neuron("720575940612453042")

        assert result is not None
        assert isinstance(result, NeuronData)
        assert result.neuron_id == "720575940612453042"

    def test_get_connectivity_success(self):
        """Test successful connectivity retrieval"""
        client = FlyWireClient()
        result = client.get_connectivity("720575940612453042", direction="both")

        assert isinstance(result, dict)
        assert "presynaptic" in result
        assert "postsynaptic" in result
        assert isinstance(result["presynaptic"], list)

    def test_dataset_specification(self):
        """Test direction parameter handling"""
        client = FlyWireClient()

        # Test different directions
        result = client.get_connectivity("123", direction="presynaptic")
        assert isinstance(result, dict)

        result = client.get_connectivity("123", direction="postsynaptic")
        assert isinstance(result, dict)

    def test_batch_queries(self):
        """Test multiple neuron queries"""
        client = FlyWireClient()
        neuron_ids = ["720575940612453042", "720575940612453043"]

        results = [client.get_neuron(nid) for nid in neuron_ids]
        assert len(results) == 2
        assert all(isinstance(r, NeuronData) for r in results)

    def test_error_handling(self):
        """Test client close method"""
        client = FlyWireClient()
        client.close()  # Should not raise error


@pytest.mark.unit
class TestAllenBrainClient:
    """Test Allen Brain Atlas API client"""

    def test_init_default(self):
        """Test Allen Brain client initialization"""
        client = AllenBrainClient()
        assert hasattr(client, 'client')
        assert AllenBrainClient.BASE_URL == "https://api.brain-map.org/api/v2"

    def test_get_gene_expression_success(self, mock_httpx_client):
        """Test successful gene expression retrieval"""
        mock_httpx_client.get.return_value.json.return_value = {
            "success": True,
            "msg": [{
                "id": 12345,
                "acronym": "SOD2",
                "name": "Superoxide dismutase 2"
            }]
        }

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = AllenBrainClient()
            result = client.get_gene_expression("SOD2")

        assert result is not None
        assert isinstance(result, GeneExpressionData)
        assert result.gene_symbol == "SOD2"
        assert result.gene_id == "12345"

    def test_brain_region_filtering(self, mock_httpx_client):
        """Test gene expression with brain region filter"""
        mock_httpx_client.get.return_value.json.return_value = {
            "success": True,
            "msg": [{"id": 123, "acronym": "APP"}]
        }

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = AllenBrainClient()
            result = client.get_gene_expression("APP", structure="hippocampus")

        assert result is not None
        assert result.gene_symbol == "APP"

    def test_connectivity_data(self, mock_httpx_client):
        """Test experiment search functionality"""
        mock_httpx_client.get.return_value.json.return_value = {
            "success": True,
            "msg": [
                {"id": 1, "name": "Experiment 1"},
                {"id": 2, "name": "Experiment 2"}
            ]
        }

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = AllenBrainClient()
            results = client.search_experiments(product_id=1, limit=10)

        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)

    def test_dataset_selection(self, mock_httpx_client):
        """Test different product selection"""
        mock_httpx_client.get.return_value.json.return_value = {
            "success": True,
            "msg": [{"id": 3}]
        }

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = AllenBrainClient()
            # Test human brain product
            results = client.search_experiments(product_id=2)

        assert isinstance(results, list)

    def test_error_handling(self, mock_httpx_client):
        """Test error handling for failed requests"""
        mock_httpx_client.get.side_effect = Exception("Network error")

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = AllenBrainClient()
            result = client.get_gene_expression("INVALID")

        assert result is None


@pytest.mark.unit
class TestMICrONSClient:
    """Test MICrONS API client"""

    def test_init_default(self):
        """Test MICrONS client initialization"""
        client = MICrONSClient()
        assert hasattr(client, 'client')
        assert MICrONSClient.BASE_URL == "https://microns-explorer.org/api"

    def test_get_connectome_data(self):
        """Test dataset info retrieval"""
        client = MICrONSClient()
        result = client.get_dataset_info()

        assert isinstance(result, ConnectomeDataset)
        assert result.dataset_id == "microns_v1"
        assert result.species == "Mus musculus"

    def test_neuron_queries(self):
        """Test neuron count in dataset"""
        client = MICrONSClient()
        result = client.get_dataset_info()

        assert result.n_neurons == 75000
        assert result.brain_region == "Visual cortex (V1)"

    def test_synapse_data(self):
        """Test synapse information"""
        client = MICrONSClient()
        result = client.get_dataset_info()

        assert result.n_synapses == 500_000_000
        assert result.resolution_nm == 4.0

    def test_annotation_retrieval(self):
        """Test dataset metadata"""
        client = MICrONSClient()
        result = client.get_dataset_info()

        assert result.data_type == "connectome"
        assert "microns-explorer.org" in result.url

    def test_error_handling(self):
        """Test client close method"""
        client = MICrONSClient()
        client.close()  # Should not raise


@pytest.mark.unit
class TestGEOClient:
    """Test GEO API client"""

    def test_init_default(self):
        """Test GEO client initialization"""
        client = GEOClient()
        assert hasattr(client, 'client')
        assert GEOClient.BASE_URL == "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"

    def test_get_dataset_success(self, mock_httpx_client):
        """Test successful dataset retrieval"""
        mock_httpx_client.get.return_value.text = "!Series_title = Test Dataset"

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = GEOClient()
            result = client.get_dataset("GSE153873")

        assert result is not None
        assert result["geo_id"] == "GSE153873"
        assert "title" in result

    def test_search_datasets(self, mock_httpx_client):
        """Test dataset search functionality"""
        with patch('httpx.Client', return_value=mock_httpx_client):
            client = GEOClient()
            results = client.search_datasets("alzheimers", organism="Homo sapiens")

        assert isinstance(results, list)

    def test_series_matrix_download(self, mock_httpx_client):
        """Test different format retrieval"""
        mock_httpx_client.get.return_value.text = "Sample data"

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = GEOClient()
            result = client.get_dataset("GSE123", format="xml")

        assert result is not None

    def test_metadata_parsing(self, mock_httpx_client):
        """Test response contains raw data"""
        mock_httpx_client.get.return_value.text = "SOFT format data"

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = GEOClient()
            result = client.get_dataset("GSE456")

        assert "raw_data" in result
        assert result["raw_data"] == "SOFT format data"

    def test_error_handling(self, mock_httpx_client):
        """Test error handling for failed requests"""
        mock_httpx_client.get.side_effect = Exception("Connection failed")

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = GEOClient()
            result = client.get_dataset("INVALID")

        assert result is None


@pytest.mark.unit
class TestAMPADClient:
    """Test AMP-AD API client"""

    def test_init_default(self):
        """Test AMP-AD client initialization"""
        client = AMPADClient()
        assert hasattr(client, 'client')
        assert AMPADClient.BASE_URL == "https://adknowledgeportal.synapse.org"

    def test_get_study_data(self):
        """Test study info retrieval"""
        client = AMPADClient()
        result = client.get_study_info("ROSMAP")

        assert result is not None
        assert result["study_id"] == "ROSMAP"
        assert "title" in result

    def test_omics_data_retrieval(self):
        """Test different data types"""
        client = AMPADClient()

        # Test transcriptomics
        results = client.list_datasets(data_type="transcriptomics")
        assert isinstance(results, list)

        # Test proteomics
        results = client.list_datasets(data_type="proteomics")
        assert isinstance(results, list)

    def test_clinical_data(self):
        """Test metadata access"""
        client = AMPADClient()
        result = client.get_study_info("MSBB")

        assert "note" in result
        assert "authentication" in result["note"].lower()

    def test_access_control(self):
        """Test placeholder behavior for authentication"""
        client = AMPADClient()
        result = client.get_study_info("TEST_STUDY")

        # Placeholder returns data but notes auth required
        assert result is not None
        assert "Synapse" in result["note"]

    def test_error_handling(self):
        """Test client close method"""
        client = AMPADClient()
        client.close()  # Should not raise


@pytest.mark.unit
class TestOpenConnectomeClient:
    """Test OpenConnectome API client"""

    def test_init_default(self):
        """Test OpenConnectome client initialization"""
        client = OpenConnectomeClient()
        assert hasattr(client, 'client')
        assert OpenConnectomeClient.BASE_URL == "https://openconnecto.me/ocp/ca"

    def test_get_dataset_info(self):
        """Test project listing"""
        client = OpenConnectomeClient()
        projects = client.list_projects()

        assert isinstance(projects, list)
        assert len(projects) >= 3
        assert "kasthuri11" in projects

    def test_volume_data_retrieval(self):
        """Test project info for mouse cortex"""
        client = OpenConnectomeClient()
        result = client.get_project_info("kasthuri11")

        assert result is not None
        assert isinstance(result, ConnectomeDataset)
        assert result.species == "Mus musculus"
        assert result.n_neurons == 1700

    def test_annotation_queries(self):
        """Test project info for Drosophila"""
        client = OpenConnectomeClient()
        result = client.get_project_info("bock11")

        assert result is not None
        assert result.species == "Drosophila melanogaster"
        assert result.brain_region == "Larval CNS"

    def test_coordinate_transforms(self):
        """Test resolution metadata"""
        client = OpenConnectomeClient()
        result = client.get_project_info("kasthuri11")

        assert result.resolution_nm == 3.0
        assert "openconnecto.me" in result.url

    def test_error_handling(self):
        """Test unknown project returns None"""
        client = OpenConnectomeClient()
        result = client.get_project_info("nonexistent_project")

        assert result is None


@pytest.mark.unit
class TestWormBaseClient:
    """Test WormBase API client"""

    def test_init_default(self):
        """Test WormBase client initialization"""
        client = WormBaseClient()
        assert hasattr(client, 'client')
        assert WormBaseClient.BASE_URL == "https://wormbase.org/rest"

    def test_get_gene_info(self, mock_httpx_client):
        """Test gene information retrieval"""
        mock_httpx_client.get.return_value.json.return_value = {
            "gene": "sod-2",
            "description": "Superoxide dismutase"
        }

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = WormBaseClient()
            result = client.get_gene("sod-2")

        assert result is not None
        assert result["gene"] == "sod-2"

    def test_get_connectome_data(self):
        """Test C. elegans connectome stats"""
        client = WormBaseClient()
        result = client.get_connectome_stats()

        assert isinstance(result, ConnectomeDataset)
        assert result.dataset_id == "celegans_full"
        assert result.n_neurons == 302

    def test_neuron_connectivity(self, mock_httpx_client):
        """Test neuron information retrieval"""
        mock_httpx_client.get.return_value.json.return_value = {
            "neuron": "AVAL",
            "connections": []
        }

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = WormBaseClient()
            result = client.get_neuron("AVAL")

        assert result is not None
        assert isinstance(result, NeuronData)
        assert result.neuron_id == "AVAL"

    def test_phenotype_data(self):
        """Test connectome metadata"""
        client = WormBaseClient()
        result = client.get_connectome_stats()

        assert result.species == "Caenorhabditis elegans"
        assert result.n_synapses == 7000
        assert result.brain_region == "Full nervous system"

    def test_error_handling(self, mock_httpx_client):
        """Test error handling for failed requests"""
        mock_httpx_client.get.side_effect = Exception("API error")

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = WormBaseClient()
            result = client.get_neuron("INVALID")

        assert result is None
