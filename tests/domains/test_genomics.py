"""
Tests for the genomics module.
"""

import pytest
from prism.domains.genomics import SequenceData, read_fastq, read_fasta, align, call_variants


def test_sequence_data():
    """Test SequenceData class."""
    sequences = ["ACGT", "GCTA"]
    ids = ["seq1", "seq2"]
    qualities = ["!!!!!", "!!!!!"]
    
    data = SequenceData(sequences=sequences, ids=ids, qualities=qualities)
    
    assert len(data) == 2
    assert data.sequences == sequences
    assert data.ids == ids
    assert data.qualities == qualities


def test_align():
    """Test align function."""
    query = SequenceData(sequences=["ACGT"], ids=["query1"])
    reference = SequenceData(sequences=["ACGTACGT"], ids=["ref1"])
    
    alignments = align(query, reference)
    
    assert isinstance(alignments, dict)
    assert "alignments" in alignments