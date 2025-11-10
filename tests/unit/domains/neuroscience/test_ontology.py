"""
Unit tests for NeuroscienceOntology (Phase 9).

Tests neuroscience domain knowledge:
- Brain region hierarchy (cortex, hippocampus, amygdala, etc.)
- Cell types (neurons, glia)
- Neurotransmitter systems
- Neurodegenerative diseases
- Disease associations (genes, regions, neurotransmitters)

Coverage target: 20 tests across 4 test classes
"""

import pytest
from kosmos.domains.neuroscience.ontology import NeuroscienceOntology
from kosmos.domains.biology.ontology import BiologicalRelationType


@pytest.fixture
def neuroscience_ontology():
    """Fixture providing a NeuroscienceOntology instance"""
    return NeuroscienceOntology()


@pytest.mark.unit
class TestNeuroscienceOntologyInit:
    """Test initialization and basic structure"""

    def test_initialization_creates_concepts(self, neuroscience_ontology):
        """Test that initialization creates all core concepts"""
        assert len(neuroscience_ontology.concepts) > 40  # 40+ core concepts

        # Verify key concepts exist
        assert "brain" in neuroscience_ontology.concepts
        assert "cortex" in neuroscience_ontology.concepts
        assert "hippocampus" in neuroscience_ontology.concepts
        assert "neuron" in neuroscience_ontology.concepts
        assert "glia" in neuroscience_ontology.concepts
        assert "dopamine" in neuroscience_ontology.concepts
        assert "alzheimers_disease" in neuroscience_ontology.concepts

    def test_concept_count_validation(self, neuroscience_ontology):
        """Test expected number of concepts in each category"""
        # Brain regions (7 major + 5 cortical = 12) + brain itself = 13
        brain_regions = [c for c in neuroscience_ontology.concepts.values()
                        if c.type in ["brain_region", "cortical_region", "anatomical_structure"]]
        assert len(brain_regions) >= 12

        # Neurons (6 subtypes) + glia (3 subtypes) + base types (2) = 11
        cell_types = [c for c in neuroscience_ontology.concepts.values()
                     if c.type in ["cell_type", "neuron_subtype", "glial_subtype"]]
        assert len(cell_types) >= 11

        # Neurotransmitters (6)
        neurotransmitters = [c for c in neuroscience_ontology.concepts.values()
                            if c.type == "neurotransmitter"]
        assert len(neurotransmitters) == 6

        # Diseases (5)
        diseases = neuroscience_ontology.get_diseases()
        assert len(diseases) == 5

    def test_relations_created(self, neuroscience_ontology):
        """Test that relationships are properly established"""
        # Should have many relations (39 total)
        assert len(neuroscience_ontology.relations) >= 35

        # Verify different relation types exist
        relation_types = {r.relation_type for r in neuroscience_ontology.relations}
        assert BiologicalRelationType.PART_OF in relation_types
        assert BiologicalRelationType.IS_A in relation_types
        assert BiologicalRelationType.ASSOCIATED_WITH in relation_types
        assert BiologicalRelationType.ENCODES in relation_types

    def test_hierarchical_structure(self, neuroscience_ontology):
        """Test brain region hierarchy construction"""
        # Get cortex hierarchy
        cortex_hierarchy = neuroscience_ontology.get_region_hierarchy("cortex")

        assert cortex_hierarchy['id'] == "cortex"
        assert cortex_hierarchy['name'] == "Cerebral Cortex"
        assert len(cortex_hierarchy['children']) == 5  # 5 cortical subregions

        # Verify subregions
        subregion_names = {child['name'] for child in cortex_hierarchy['children']}
        assert "Prefrontal Cortex" in subregion_names
        assert "Motor Cortex" in subregion_names
        assert "Visual Cortex" in subregion_names


@pytest.mark.unit
class TestBrainRegions:
    """Test brain region concepts and relationships"""

    def test_brain_region_hierarchy(self, neuroscience_ontology):
        """Test major brain regions are children of brain"""
        brain_regions = neuroscience_ontology.get_child_concepts("brain", BiologicalRelationType.PART_OF)

        # Should have 7 major regions
        assert len(brain_regions) >= 7

        region_names = {r.name for r in brain_regions}
        assert "Cerebral Cortex" in region_names
        assert "Hippocampus" in region_names
        assert "Basal Ganglia" in region_names
        assert "Thalamus" in region_names
        assert "Cerebellum" in region_names

    def test_cortex_subregions(self, neuroscience_ontology):
        """Test cortical subregions have correct properties"""
        # Get cortical subregions
        cortical_regions = neuroscience_ontology.get_child_concepts("cortex", BiologicalRelationType.PART_OF)
        assert len(cortical_regions) == 5

        # Verify prefrontal cortex
        assert "prefrontal_cortex" in neuroscience_ontology.concepts
        pfc = neuroscience_ontology.concepts["prefrontal_cortex"]
        assert pfc.name == "Prefrontal Cortex"
        assert pfc.type == "cortical_region"
        assert "executive function" in pfc.description.lower()

    def test_subcortical_structures(self, neuroscience_ontology):
        """Test subcortical brain structures"""
        # Hippocampus
        assert "hippocampus" in neuroscience_ontology.concepts
        hippocampus = neuroscience_ontology.concepts["hippocampus"]
        assert hippocampus.name == "Hippocampus"
        assert hippocampus.type == "brain_region"
        assert "UBERON" in hippocampus.external_ids
        assert hippocampus.external_ids["UBERON"] == "0002421"

        # Amygdala
        assert "amygdala" in neuroscience_ontology.concepts
        amygdala = neuroscience_ontology.concepts["amygdala"]
        assert amygdala.name == "Amygdala"
        assert "temporal lobe" in amygdala.description.lower()

    def test_region_connectivity(self, neuroscience_ontology):
        """Test that regions have parent relationships"""
        # Check cortical region parents
        pfc_parents = neuroscience_ontology.get_parent_concepts(
            "prefrontal_cortex",
            BiologicalRelationType.PART_OF
        )
        assert len(pfc_parents) == 1
        assert pfc_parents[0].id == "cortex"

        # Check cortex parents
        cortex_parents = neuroscience_ontology.get_parent_concepts(
            "cortex",
            BiologicalRelationType.PART_OF
        )
        assert len(cortex_parents) == 1
        assert cortex_parents[0].id == "brain"

    def test_functional_areas(self, neuroscience_ontology):
        """Test functional brain areas have correct descriptions"""
        # Motor cortex
        motor = neuroscience_ontology.concepts["motor_cortex"]
        assert "motor" in motor.description.lower() or "movement" in motor.description.lower()

        # Visual cortex
        visual = neuroscience_ontology.concepts["visual_cortex"]
        assert "visual" in visual.description.lower()

        # Temporal cortex
        temporal = neuroscience_ontology.concepts["temporal_cortex"]
        assert "auditory" in temporal.description.lower() or "memory" in temporal.description.lower()


@pytest.mark.unit
class TestCellTypes:
    """Test neuron and glial cell type concepts"""

    def test_neuron_types(self, neuroscience_ontology):
        """Test neuron subtypes and hierarchy"""
        # Get all neuron subtypes
        neuron_subtypes = neuroscience_ontology.get_child_concepts("neuron", BiologicalRelationType.IS_A)
        assert len(neuron_subtypes) == 6

        # Verify specific neuron types
        neuron_names = {n.name for n in neuron_subtypes}
        assert "Pyramidal Neuron" in neuron_names
        assert "Dopaminergic Neuron" in neuron_names
        assert "GABAergic Neuron" in neuron_names
        assert "Glutamatergic Neuron" in neuron_names
        assert "Motor Neuron" in neuron_names
        assert "Interneuron" in neuron_names

    def test_glial_cells(self, neuroscience_ontology):
        """Test glial cell types"""
        # Get glial subtypes
        glia_subtypes = neuroscience_ontology.get_child_concepts("glia", BiologicalRelationType.IS_A)
        assert len(glia_subtypes) == 3

        # Verify specific glial types
        glia_names = {g.name for g in glia_subtypes}
        assert "Astrocyte" in glia_names
        assert "Microglia" in glia_names
        assert "Oligodendrocyte" in glia_names

        # Check astrocyte details
        astrocyte = neuroscience_ontology.concepts["astrocyte"]
        assert astrocyte.type == "glial_subtype"
        assert "support" in astrocyte.description.lower() or "star" in astrocyte.description.lower()

    def test_cell_type_hierarchy(self, neuroscience_ontology):
        """Test cell type hierarchical relationships"""
        # Pyramidal neurons should be child of neuron
        pyramidal_parents = neuroscience_ontology.get_parent_concepts(
            "pyramidal_neuron",
            BiologicalRelationType.IS_A
        )
        assert len(pyramidal_parents) == 1
        assert pyramidal_parents[0].id == "neuron"

        # Microglia should be child of glia
        microglia_parents = neuroscience_ontology.get_parent_concepts(
            "microglia",
            BiologicalRelationType.IS_A
        )
        assert len(microglia_parents) == 1
        assert microglia_parents[0].id == "glia"

    def test_cell_markers(self, neuroscience_ontology):
        """Test cell type descriptions contain key markers/characteristics"""
        # Dopaminergic neurons produce dopamine
        dopaminergic = neuroscience_ontology.concepts["dopaminergic_neuron"]
        assert "dopamine" in dopaminergic.description.lower()

        # GABAergic neurons use GABA
        gabaergic = neuroscience_ontology.concepts["gabaergic_neuron"]
        assert "gaba" in gabaergic.description.lower()

        # Microglia are immune cells
        microglia = neuroscience_ontology.concepts["microglia"]
        assert "immune" in microglia.description.lower()


@pytest.mark.unit
class TestNeurotransmitters:
    """Test neurotransmitter systems"""

    def test_neurotransmitter_systems(self, neuroscience_ontology):
        """Test all major neurotransmitters are present"""
        # Get all neurotransmitters
        neurotransmitters = [c for c in neuroscience_ontology.concepts.values()
                            if c.type == "neurotransmitter"]
        assert len(neurotransmitters) == 6

        nt_names = {nt.name for nt in neurotransmitters}
        assert "Dopamine" in nt_names
        assert "Serotonin" in nt_names
        assert "Glutamate" in nt_names
        assert "GABA" in nt_names
        assert "Acetylcholine" in nt_names
        assert "Norepinephrine" in nt_names

    def test_receptor_associations(self, neuroscience_ontology):
        """Test neurotransmitter-neuron associations"""
        # Dopaminergic neurons encode dopamine
        # Find ENCODES relations where target is dopamine
        dopamine_relations = [
            r for r in neuroscience_ontology.relations
            if r.target_id == "dopamine" and r.relation_type == BiologicalRelationType.ENCODES
        ]
        assert len(dopamine_relations) == 1
        assert dopamine_relations[0].source_id == "dopaminergic_neuron"

        # GABAergic neurons encode GABA
        gaba_relations = [
            r for r in neuroscience_ontology.relations
            if r.target_id == "gaba" and r.relation_type == BiologicalRelationType.ENCODES
        ]
        assert len(gaba_relations) == 1
        assert gaba_relations[0].source_id == "gabaergic_neuron"

    def test_synaptic_transmission(self, neuroscience_ontology):
        """Test synaptic transmission process"""
        assert "synaptic_transmission" in neuroscience_ontology.concepts
        synaptic = neuroscience_ontology.concepts["synaptic_transmission"]

        assert synaptic.name == "Synaptic Transmission"
        assert synaptic.type == "biological_process"
        assert "synapse" in synaptic.description.lower()
        assert "GO" in synaptic.external_ids

    def test_signaling_pathways(self, neuroscience_ontology):
        """Test neurotransmitter chemical IDs are present"""
        # Dopamine has CHEBI ID
        dopamine = neuroscience_ontology.concepts["dopamine"]
        assert "CHEBI" in dopamine.external_ids
        assert dopamine.external_ids["CHEBI"] == "18243"

        # Serotonin has CHEBI ID
        serotonin = neuroscience_ontology.concepts["serotonin"]
        assert "CHEBI" in serotonin.external_ids
        assert dopamine.description.lower() == "catecholamine neurotransmitter"


@pytest.mark.unit
class TestDiseaseConcepts:
    """Test neurodegenerative disease concepts and associations"""

    def test_neurodegenerative_diseases(self, neuroscience_ontology):
        """Test all major neurodegenerative diseases are present"""
        diseases = neuroscience_ontology.get_diseases()
        assert len(diseases) == 5

        disease_names = {d.name for d in diseases}
        assert "Alzheimer's Disease" in disease_names
        assert "Parkinson's Disease" in disease_names
        assert "Huntington's Disease" in disease_names
        assert "Amyotrophic Lateral Sclerosis" in disease_names
        assert "Multiple Sclerosis" in disease_names

        # Check Alzheimer's details
        ad = neuroscience_ontology.concepts["alzheimers_disease"]
        assert ad.type == "disease"
        assert "DOID" in ad.external_ids
        assert "memory" in ad.description.lower()

    def test_disease_brain_region_associations(self, neuroscience_ontology):
        """Test disease-brain region associations"""
        # Alzheimer's affects hippocampus and cortex
        ad_regions = neuroscience_ontology.get_disease_regions("alzheimers_disease")
        assert len(ad_regions) == 2
        region_names = {r.name for r in ad_regions}
        assert "Hippocampus" in region_names
        assert "Cerebral Cortex" in region_names

        # Parkinson's affects basal ganglia
        pd_regions = neuroscience_ontology.get_disease_regions("parkinsons_disease")
        assert len(pd_regions) == 1
        assert pd_regions[0].name == "Basal Ganglia"

    def test_disease_progression(self, neuroscience_ontology):
        """Test disease-gene associations"""
        # Alzheimer's disease genes
        ad_genes = neuroscience_ontology.get_disease_genes("alzheimers_disease")
        assert len(ad_genes) == 4

        gene_symbols = {g.name for g in ad_genes}
        assert "Amyloid Precursor Protein" in gene_symbols
        assert "Apolipoprotein E" in gene_symbols
        assert "Presenilin 1" in gene_symbols
        assert "Microtubule-Associated Protein Tau" in gene_symbols

        # Check APP details
        app_concept = next(g for g in ad_genes if "Amyloid" in g.name)
        assert app_concept.id == "APP"
        assert app_concept.type == "gene"
        assert "HGNC" in app_concept.external_ids
