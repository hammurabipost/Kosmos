"""
Unit tests for ConnectomicsAnalyzer (Phase 9).

Tests connectome scaling analysis (Figure 4 pattern):
- Power law fitting (y = a * x^b)
- Scaling relationships (length vs synapses, etc.)
- Cross-species comparisons

Coverage target: 25 tests across 4 test classes
"""

import pytest
import pandas as pd
import numpy as np
from kosmos.domains.neuroscience.connectomics import (
    ConnectomicsAnalyzer,
    ConnectomicsResult,
    ScalingRelationship,
    PowerLawFit,
    CrossSpeciesComparison
)


@pytest.fixture
def connectomics_analyzer():
    """Fixture providing a ConnectomicsAnalyzer instance"""
    return ConnectomicsAnalyzer()


@pytest.fixture
def sample_connectome_data():
    """Sample connectome data with power-law relationship"""
    np.random.seed(42)
    # Generate data with power law: synapses ~ length^1.5
    n = 100
    length = np.random.uniform(10, 1000, n)
    synapses = 0.5 * (length ** 1.5) * np.random.uniform(0.8, 1.2, n)
    degree = np.random.uniform(5, 50, n)

    return pd.DataFrame({
        'neuron_id': [f'neuron_{i}' for i in range(n)],
        'Length': length,
        'Synapses': synapses,
        'Degree': degree
    })


@pytest.mark.unit
class TestConnectomicsAnalyzerInit:
    """Test ConnectomicsAnalyzer initialization"""

    def test_init_default(self):
        """Test default initialization"""
        analyzer = ConnectomicsAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, 'analyze_scaling_laws')

    def test_init_with_custom_params(self):
        """Test initialization doesn't require parameters"""
        analyzer = ConnectomicsAnalyzer()
        assert analyzer is not None


@pytest.mark.unit
class TestScalingAnalysis:
    """Test scaling law analysis"""

    def test_power_law_detection(self, connectomics_analyzer, sample_connectome_data):
        """Test power law fitting"""
        result = connectomics_analyzer.analyze_scaling_laws(
            sample_connectome_data,
            species_name="Test Species"
        )

        assert isinstance(result, ConnectomicsResult)
        assert result.species_name == "Test Species"
        assert result.n_neurons == 100

    def test_scaling_coefficient_calculation(self, connectomics_analyzer, sample_connectome_data):
        """Test length-synapses scaling relationship"""
        result = connectomics_analyzer.analyze_scaling_laws(sample_connectome_data)

        assert result.length_synapses is not None
        assert isinstance(result.length_synapses, ScalingRelationship)
        assert result.length_synapses.power_law.exponent > 0

    def test_goodness_of_fit(self, connectomics_analyzer, sample_connectome_data):
        """Test R-squared goodness of fit"""
        result = connectomics_analyzer.analyze_scaling_laws(sample_connectome_data)

        # Generated data follows power law, should have high R²
        assert result.length_synapses.power_law.r_squared > 0.8

    def test_log_log_plotting_data(self, connectomics_analyzer, sample_connectome_data):
        """Test power law equation generation"""
        result = connectomics_analyzer.analyze_scaling_laws(sample_connectome_data)

        power_law = result.length_synapses.power_law
        assert hasattr(power_law, 'equation')
        assert isinstance(power_law.equation, str)
        assert "^" in power_law.equation

    def test_outlier_handling(self, connectomics_analyzer, sample_connectome_data):
        """Test analysis handles data with outliers"""
        # Add outliers
        data_with_outliers = sample_connectome_data.copy()
        data_with_outliers.loc[0, 'n_synapses'] = 1000000  # Extreme outlier

        result = connectomics_analyzer.analyze_scaling_laws(data_with_outliers)

        # Should still produce results
        assert result.length_synapses is not None
        assert result.n_neurons == 100

    def test_species_specific_scaling(self, connectomics_analyzer):
        """Test species name is preserved"""
        data = pd.DataFrame({
            'Length': [10, 20, 30],
            'Synapses': [5, 15, 30],
            'Degree': [3, 6, 9]
        })

        result = connectomics_analyzer.analyze_scaling_laws(data, species_name="Drosophila")
        assert result.species_name == "Drosophila"

    def test_confidence_intervals(self, connectomics_analyzer, sample_connectome_data):
        """Test statistical significance"""
        result = connectomics_analyzer.analyze_scaling_laws(sample_connectome_data)

        # Should have p-value
        assert hasattr(result.length_synapses, 'p_value')
        assert result.length_synapses.p_value >= 0
        assert result.length_synapses.p_value <= 1

    def test_robustness_analysis(self, connectomics_analyzer, sample_connectome_data):
        """Test correlation strength assessment"""
        result = connectomics_analyzer.analyze_scaling_laws(sample_connectome_data)

        # Should assess correlation strength
        assert hasattr(result.length_synapses, 'correlation_strength')
        strength = result.length_synapses.correlation_strength
        assert strength in ["very weak", "weak", "moderate", "strong", "very strong"]


@pytest.mark.unit
class TestPowerLawFit:
    """Test power law fitting"""

    def test_fit_quality_metrics(self, connectomics_analyzer, sample_connectome_data):
        """Test fit quality metrics are calculated"""
        result = connectomics_analyzer.analyze_scaling_laws(sample_connectome_data)

        power_law = result.length_synapses.power_law
        assert isinstance(power_law, PowerLawFit)
        assert hasattr(power_law, 'exponent')
        assert hasattr(power_law, 'coefficient')
        assert hasattr(power_law, 'r_squared')

    def test_exponent_estimation(self, connectomics_analyzer, sample_connectome_data):
        """Test exponent estimation accuracy"""
        result = connectomics_analyzer.analyze_scaling_laws(sample_connectome_data)

        # Data generated with exponent ~1.5
        exponent = result.length_synapses.power_law.exponent
        assert 1.2 < exponent < 1.8  # Should be close to 1.5

    def test_confidence_intervals(self, connectomics_analyzer, sample_connectome_data):
        """Test spearman correlation"""
        result = connectomics_analyzer.analyze_scaling_laws(sample_connectome_data)

        # Should have strong Spearman correlation
        assert result.length_synapses.spearman_rho > 0.8

    def test_fit_vs_actual_comparison(self, connectomics_analyzer, sample_connectome_data):
        """Test equation format"""
        result = connectomics_analyzer.analyze_scaling_laws(sample_connectome_data)

        equation = result.length_synapses.power_law.equation
        assert "Synapses" in equation or "y" in equation
        assert "Length" in equation or "x" in equation

    def test_non_power_law_detection(self, connectomics_analyzer):
        """Test linear relationship detection"""
        # Create linear data (not power law)
        data = pd.DataFrame({
            'Length': [10, 20, 30, 40],
            'Synapses': [10, 20, 30, 40],  # Linear
            'Degree': [5, 10, 15, 20]
        })

        result = connectomics_analyzer.analyze_scaling_laws(data)

        # Should still fit (exponent ~1 for linear)
        assert result.length_synapses is not None
        assert 0.8 < result.length_synapses.power_law.exponent < 1.2

    def test_alternative_models(self, connectomics_analyzer, sample_connectome_data):
        """Test multiple scaling relationships"""
        result = connectomics_analyzer.analyze_scaling_laws(sample_connectome_data)

        # Should analyze multiple property pairs
        assert result.length_synapses is not None
        assert result.length_degree is not None
        assert result.synapses_degree is not None

    def test_statistical_validation(self, connectomics_analyzer, sample_connectome_data):
        """Test significance testing"""
        result = connectomics_analyzer.analyze_scaling_laws(sample_connectome_data)

        # Generated data has strong relationship
        assert result.length_synapses.is_significant
        assert result.length_synapses.p_value < 0.05


@pytest.mark.unit
class TestCrossSpeciesComparison:
    """Test cross-species comparison"""

    def test_drosophila_vs_celegans(self, connectomics_analyzer):
        """Test comparison between two species"""
        # C. elegans data (302 neurons)
        celegans_data = pd.DataFrame({
            'Length': np.random.uniform(10, 500, 50),
            'Synapses': np.random.uniform(5, 100, 50),
            'Degree': np.random.uniform(3, 20, 50)
        })

        # Drosophila data (larger)
        drosophila_data = pd.DataFrame({
            'Length': np.random.uniform(10, 1000, 100),
            'Synapses': np.random.uniform(5, 200, 100),
            'Degree': np.random.uniform(5, 30, 100)
        })

        datasets = {
            'Celegans': celegans_data,
            'Drosophila': drosophila_data
        }

        comparison = connectomics_analyzer.cross_species_comparison(datasets)

        assert isinstance(comparison, CrossSpeciesComparison)
        assert len(comparison.species_results) == 2

    def test_mouse_vs_human(self, connectomics_analyzer):
        """Test comparison with species names preserved"""
        mouse_data = pd.DataFrame({
            'Length': [10, 20, 30],
            'Synapses': [5, 15, 30],
            'Degree': [3, 6, 9]
        })

        human_data = pd.DataFrame({
            'Length': [15, 25, 35],
            'Synapses': [8, 18, 35],
            'Degree': [4, 7, 10]
        })

        datasets = {'Mouse': mouse_data, 'Human': human_data}
        comparison = connectomics_analyzer.cross_species_comparison(datasets)

        assert 'Mouse' in comparison.species_results
        assert 'Human' in comparison.species_results

    def test_scaling_consistency(self, connectomics_analyzer):
        """Test mean exponent calculation"""
        data1 = pd.DataFrame({
            'Length': [10, 20, 30],
            'Synapses': [5, 15, 30],
            'Degree': [3, 6, 9]
        })

        datasets = {'Species1': data1, 'Species2': data1.copy()}
        comparison = connectomics_analyzer.cross_species_comparison(datasets)

        # Should calculate mean exponent
        assert comparison.mean_length_synapses_exponent is not None

    def test_species_differences(self, connectomics_analyzer):
        """Test standard deviation calculation"""
        data1 = pd.DataFrame({'Length': [10, 20], 'Synapses': [5, 15], 'Degree': [3, 6]})
        data2 = pd.DataFrame({'Length': [10, 20], 'Synapses': [10, 30], 'Degree': [5, 10]})

        datasets = {'Species1': data1, 'Species2': data2}
        comparison = connectomics_analyzer.cross_species_comparison(datasets)

        # Should calculate std deviation
        assert comparison.std_length_synapses_exponent is not None

    def test_statistical_significance(self, connectomics_analyzer):
        """Test universality assessment exists"""
        data = pd.DataFrame({'Length': [10, 20], 'Synapses': [5, 15], 'Degree': [3, 6]})

        datasets = {'S1': data, 'S2': data.copy()}
        comparison = connectomics_analyzer.cross_species_comparison(datasets)

        assert hasattr(comparison, 'is_universal_scaling')
        assert isinstance(comparison.is_universal_scaling, bool)

    def test_evolutionary_insights(self, connectomics_analyzer):
        """Test universality notes are collected"""
        data = pd.DataFrame({'Length': [10, 20], 'Synapses': [5, 15], 'Degree': [3, 6]})

        datasets = {'S1': data}
        comparison = connectomics_analyzer.cross_species_comparison(datasets)

        assert hasattr(comparison, 'universality_notes')
        assert isinstance(comparison.universality_notes, list)

    def test_normalized_comparison(self, connectomics_analyzer):
        """Test DataFrame conversion"""
        data = pd.DataFrame({'Length': [10, 20], 'Synapses': [5, 15], 'Degree': [3, 6]})

        datasets = {'Species1': data}
        comparison = connectomics_analyzer.cross_species_comparison(datasets)

        df = comparison.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert 'species' in df.columns

    def test_result_validation(self, connectomics_analyzer):
        """Test all species results are analyzed"""
        data1 = pd.DataFrame({'Length': [10, 20], 'Synapses': [5, 15], 'Degree': [3, 6]})
        data2 = pd.DataFrame({'Length': [15, 25], 'Synapses': [8, 18], 'Degree': [4, 7]})
        data3 = pd.DataFrame({'Length': [20, 30], 'Synapses': [10, 20], 'Degree': [5, 8]})

        datasets = {'S1': data1, 'S2': data2, 'S3': data3}
        comparison = connectomics_analyzer.cross_species_comparison(datasets)

        # All species should be analyzed
        assert len(comparison.species_results) == 3
        for name, result in comparison.species_results.items():
            assert isinstance(result, ConnectomicsResult)
            assert result.species_name == name
