import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, VarianceThreshold

from har_subject_shift.data import LABELS, verify_zip, atomic_json
from har_subject_shift.metrics import score, subject_scores, cluster_bootstrap, family_errors
from har_subject_shift.experiment import fold_invariant, outer_group_predictions
from har_subject_shift.models import candidates, selection_pipeline


def small_data():
    rng = np.random.default_rng(42)
    subjects = np.repeat(np.arange(1, 11), 12)
    y = np.tile(np.arange(1, 7), 20).astype(int)
    x = rng.normal(size=(len(y), 15)).astype('float32')
    x[:, :6] += np.eye(6)[y-1] * 3
    return x, y, subjects


def test_six_activity_schema():
    assert len(LABELS) == 6
    assert set(LABELS) == set(range(1, 7))
    assert len(candidates('quick')) == 6


def test_subject_leakage_guard():
    groups = np.array([1, 1, 2, 2, 3, 3])
    assert fold_invariant(groups, np.array([0,1,2,3]), np.array([4,5]))
    with pytest.raises(AssertionError):
        fold_invariant(groups, np.array([0,2]), np.array([1,3]))


def test_oof_exactly_once_and_reproducible():
    x,y,g = small_data()
    model = Pipeline([('model', DummyClassifier(strategy='most_frequent'))])
    p, folds, scores, manifest = outer_group_predictions(model, x, y, g, outer_splits=5, inner_splits=3)
    p2, folds2, _, _ = outer_group_predictions(model, x, y, g, outer_splits=5, inner_splits=3)
    assert np.array_equal(p,p2)
    assert np.array_equal(folds,folds2)
    assert sorted(np.unique(folds)) == [1,2,3,4,5]
    assert len(p)==len(y) and len(scores)==5 and len(manifest)==5
    for item in manifest:
        assert not set(item['train_subjects']) & set(item['validation_subjects'])


def test_nested_group_cv_and_pipeline():
    x,y,g = small_data()
    model, grid = candidates('quick')['knn']
    pred, folds, records, _ = outer_group_predictions(model,x,y,g,grid=grid)
    assert len(pred)==len(y)
    assert all('params' in record for record in records)
    select = selection_pipeline(4)
    assert any(isinstance(step,SelectKBest) for _, step in select.steps)
    assert isinstance(select.steps[0][1],VarianceThreshold)


def test_metrics_fixed_classes_confusion_and_absent_classes():
    y=np.array([1,1,2,2])
    p=np.array([1,2,2,1])
    m=score(y,p)
    cm=np.asarray(m['confusion'])
    assert cm.shape==(6,6) and cm.sum()==len(y)
    assert len(m['absent_true_classes'])==4
    assert np.isclose(m['balanced_accuracy'], (0.5+0.5)/6)
    f=family_errors(y,p)
    assert f['dynamic']['n_windows']==4
    assert f['static']['n_windows']==0


def test_bootstrap_resamples_subjects_not_windows():
    y=np.array([1,1,2,2,1,1,2,2])
    p=np.array([1,2,2,2,1,1,1,2])
    g=np.array([10,10,10,10,11,11,11,11])
    a=cluster_bootstrap(y,p,g,b=80,seed=42)
    b=cluster_bootstrap(y,p,g,b=80,seed=42)
    assert a==b
    assert a['n_subjects']==2
    assert a['B']==80
    assert 0<=a['global_macro_f1_ci_95'][0]<=a['global_macro_f1_ci_95'][1]<=1


def test_zip_slip_rejected(tmp_path):
    import zipfile
    with zipfile.ZipFile(tmp_path/'bad.zip','w') as z:
        z.writestr('../evil', 'bad')
    with pytest.raises(ValueError,match='inseguro'):
        verify_zip(tmp_path/'bad.zip')


def test_holdout_requires_lock_before_any_load(monkeypatch,tmp_path):
    import har_subject_shift.experiment as e
    monkeypatch.setattr(e, 'require_audit', lambda root: True)
    def should_not_read(*_): raise AssertionError('Acesso ao holdout antes do lock')
    monkeypatch.setattr(e, 'load_split', should_not_read)
    with pytest.raises(RuntimeError,match='protocol_locked'):
        e.final_test(tmp_path)


def test_train_cannot_read_test(monkeypatch,tmp_path):
    import har_subject_shift.experiment as e
    x,y,g=small_data()
    (tmp_path/'artifacts').mkdir()
    atomic_json(tmp_path/'artifacts/data_manifest.json', {'sha256':'fixture_only_not_real_data'})
    accesses=[]
    def isolated(root,split):
        accesses.append(split)
        if split == 'test': raise AssertionError('Teste lido durante seleção!')
        return x,y,g
    monkeypatch.setattr(e, 'require_audit', lambda root: True)
    monkeypatch.setattr(e, 'load_split', isolated)
    monkeypatch.setattr(e, 'candidates', lambda mode: {'dummy': (Pipeline([('model',DummyClassifier(strategy='most_frequent'))]),{})})
    lock=e.train(tmp_path,mode='quick',run_hypotheses=False)
    assert accesses==['train']
    assert lock['test_status']=='NEVER_OPENED_FOR_MODELING'
    assert (tmp_path/'artifacts/protocol_locked.json').exists()
    with pytest.raises(RuntimeError,match='já congelado'):
        e.train(tmp_path,mode='quick',run_hypotheses=False)
