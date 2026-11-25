"""Integration tests use official ZIP only if explicitly present; never hallucinate it."""
from pathlib import Path
import numpy as np
import pytest
from har_subject_shift.data import verify_zip, extract, load_split

ROOT = Path(__file__).parents[1]
ZIP = ROOT/'data/raw/uci_har_official.zip'

@pytest.mark.skipif(not ZIP.exists(), reason='ZIP oficial não disponível neste ambiente; integração real pendente')
def test_original_data_shapes_and_disjoint_subjects():
    manifest=verify_zip(ZIP)
    extract(ROOT)
    tr=load_split(ROOT,'train')
    te=load_split(ROOT,'test')
    assert tr[0].shape==(7352,561)
    assert te[0].shape==(2947,561)
    assert not set(tr[2]) & set(te[2])
    assert len(set(tr[2]))==21 and len(set(te[2]))==9
    assert len(set(tr[1]))==len(set(te[1]))==6
    assert manifest['crc']=='PASS'
