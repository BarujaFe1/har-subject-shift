"""UCI HAR acquisition, verified extraction, and leakage-aware data loading."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

URL = "https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip"
LABELS = {1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS", 4: "SITTING", 5: "STANDING", 6: "LAYING"}
IDS = tuple(range(1, 7))
EXPECTED = {"train": 7352, "test": 2947}
# Documented in independent reproductions; confirmed only after local primary-file audit.
EXPECTED_SUBJECTS = {
    "train": {1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30},
    "test": {2, 4, 9, 10, 12, 13, 18, 20, 24},
}
EXPECTED_CLASS_COUNTS = {
    "train": {1: 1226, 2: 1073, 3: 986, 4: 1286, 5: 1374, 6: 1407},
    "test": {1: 496, 2: 471, 3: 420, 4: 491, 5: 532, 6: 537},
}
REQUIRED = ["README.txt", "features.txt", "features_info.txt", "activity_labels.txt"] + [f"{split}/{stem}_{split}.txt" for split in ("train", "test") for stem in ("X", "y", "subject")]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, content: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf8", delete=False) as f:
        json.dump(content, f, indent=2, ensure_ascii=False, default=str)
        tmp = Path(f.name)
    os.replace(tmp, path)


def download(root: Path) -> Path:
    """Download to a temporary file; never pretend a failed download is complete."""
    out = root / "data/raw/uci_har_official.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        verify_zip(out)
        return out
    tmp = out.with_suffix(".part")
    try:
        with requests.get(URL, stream=True, timeout=(15, 120), headers={"User-Agent": "har-subject-shift-research/0.1"}) as response:
            response.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
        verify_zip(tmp)
        tmp.replace(out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return out


def verify_zip(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"ZIP oficial ausente: {path}. Use 'har-shift acquire' em máquina com rede.")
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        if not infos:
            raise ValueError("Arquivo ZIP vazio")
        if sum(i.file_size for i in infos) > 2_000_000_000:
            raise ValueError("Arquivo descompactado excederia o limite de segurança de 2 GB")
        for info in infos:
            p = Path(info.filename)
            if p.is_absolute() or ".." in p.parts or "\\" in info.filename:
                raise ValueError(f"Caminho inseguro no ZIP: {info.filename}")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"ZIP contém link simbólico: {info.filename}")
            if info.compress_size and info.file_size > 500 * info.compress_size:
                raise ValueError(f"Razão de compressão suspeita: {info.filename}")
        failed = z.testzip()  # tests every entry CRC, not only zip metadata
        if failed:
            raise ValueError(f"CRC inválido: {failed}")
        names = [i.filename for i in infos if not i.is_dir()]
        expected = ["UCI HAR Dataset/" + r for r in REQUIRED]
        missing = [p for p in expected if p not in names]
        if missing:
            raise ValueError(f"ZIP sem arquivos obrigatórios: {missing}")
        return {"zip_bytes": path.stat().st_size, "sha256": sha256(path), "file_count": len(names), "uncompressed_bytes": sum(i.file_size for i in infos), "crc": "PASS"}


def extract(root: Path) -> tuple[Path, dict]:
    archive = root / "data/raw/uci_har_official.zip"
    manifest = verify_zip(archive)
    target = root / "data/extracted"
    dataset = target / "UCI HAR Dataset"
    if not all((dataset / x).is_file() for x in REQUIRED):
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
    if not all((dataset / x).is_file() for x in REQUIRED):
        raise ValueError("Extração incompleta")
    manifest.update({"dataset_url": URL, "doi": "10.24432/C54S4K", "fetched_utc": datetime.now(timezone.utc).isoformat(), "extracted_file_bytes": sum(p.stat().st_size for p in dataset.rglob("*") if p.is_file()), "source": "primary UCI", "license_dispute": "Portal CC BY 4.0; historical README disallows commercial use — unresolved"})
    return dataset, manifest


def load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Only explicitly requested split is loaded. Never append subject IDs to X."""
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    base = root / "data/extracted/UCI HAR Dataset" / split
    x = np.loadtxt(base / f"X_{split}.txt", dtype=np.float32)
    y = np.loadtxt(base / f"y_{split}.txt", dtype=np.int16, ndmin=1)
    groups = np.loadtxt(base / f"subject_{split}.txt", dtype=np.int16, ndmin=1)
    if x.ndim != 2 or x.shape != (EXPECTED[split], 561):
        raise ValueError(f"Dimensão inesperada de {split}: {x.shape}")
    if y.shape != (len(x),) or groups.shape != (len(x),):
        raise ValueError(f"Desalinhamento X/y/subject em {split}")
    if not np.isfinite(x).all():
        raise ValueError(f"NaN/Inf em {split}")
    if set(np.unique(y)) != set(IDS):
        raise ValueError(f"Conjunto de classes divergente em {split}")
    if not np.all((groups >= 1) & (groups <= 30)):
        raise ValueError("subject_id fora de 1..30")
    return x, y, groups


def audit(root: Path) -> dict:
    """Structural audit of both splits; never trains/tunes/evaluates predictions on test."""
    _, manifest = extract(root)
    base = root / "data/extracted/UCI HAR Dataset"
    features = (base / "features.txt").read_text(encoding="utf8").splitlines()
    parsed = [line.split(maxsplit=1) for line in features]
    if len(parsed) != 561 or any(len(p) != 2 or int(p[0]) != i for i, p in enumerate(parsed, 1)):
        raise ValueError("features.txt não contém posições ordinais 1..561")
    names = [p[1] for p in parsed]
    label_lines = (base / "activity_labels.txt").read_text(encoding="utf8").splitlines()
    if {int(line.split()[0]): line.split()[1] for line in label_lines} != LABELS:
        raise ValueError("Mapeamento de atividades inesperado")
    datasets, by_split = {}, {}
    for split in ("train", "test"):
        x, y, groups = load_split(root, split)
        if set(map(int, np.unique(groups))) != EXPECTED_SUBJECTS[split]:
            raise ValueError(f"IDs de participantes inesperados em {split}; investigar fonte do ZIP")
        if {k: int(np.sum(y == k)) for k in IDS} != EXPECTED_CLASS_COUNTS[split]:
            raise ValueError(f"Contagem por atividade inesperada em {split}; investigar fonte do ZIP")
        datasets[split] = (x, y, groups)
        by_split[split] = {"shape": list(x.shape), "n_subjects": len(np.unique(groups)), "subject_ids": sorted(map(int, np.unique(groups))), "class_counts": {LABELS[k]: int(np.sum(y == k)) for k in IDS}, "duplicate_rows_within_split": int(len(x) - len(np.unique(x, axis=0))), "min_value": float(x.min()), "max_value": float(x.max()), "nonfinite": int(np.sum(~np.isfinite(x)))}
    tr, te = datasets["train"], datasets["test"]
    overlap = set(tr[2]) & set(te[2])
    if overlap or len(set(tr[2])) != 21 or len(set(te[2])) != 9:
        raise AssertionError(f"Divisão entre sujeitos incorreta: {overlap}")
    # Real duplicate feature vectors, not only duplicate feature names
    tr_bytes = {hashlib.sha256(row.tobytes()).digest() for row in tr[0]}
    cross_duplicates = int(sum(hashlib.sha256(row.tobytes()).digest() in tr_bytes for row in te[0]))
    frame = pd.crosstab(pd.Series(tr[2], name="subject_id"), pd.Series(tr[1], name="activity_id"))
    frame = frame.reindex(columns=list(IDS), fill_value=0)
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / "artifacts/subject_activity_train.csv")
    manifest["readme_sha256"] = sha256(base / "README.txt")
    manifest["readme_mentions_commercial_restriction"] = "Any commercial use is prohibited" in (base / "README.txt").read_text(encoding="utf8", errors="replace")
    audit_result = {"status": "PASS", "source": "local official ZIP", "splits": by_split, "n_features": len(names), "unique_feature_names": len(set(names)), "duplicated_feature_name_positions": len(names) - len(set(names)), "canonical_feature_ids": "f001..f561", "cross_split_subject_overlap": sorted(map(int, overlap)), "cross_split_exact_duplicate_vectors": cross_duplicates, "train_activity_subject_minimum": int(frame.to_numpy().min()), "train_subjects_missing_class": int((frame == 0).any(axis=1).sum()), "feature_names_sha256": sha256(base / "features.txt"), "audit_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(root / "artifacts/data_manifest.json", manifest)
    atomic_json(root / "artifacts/dataset_audit.json", audit_result)
    return audit_result


def require_audit(root: Path) -> dict:
    audit_file = root / "artifacts/dataset_audit.json"
    if not audit_file.exists():
        raise RuntimeError("G0/G1 obrigatórios: execute 'har-shift audit' antes dos modelos")
    audit_result = json.loads(audit_file.read_text(encoding="utf8"))
    if audit_result.get("status") != "PASS":
        raise RuntimeError("Auditoria não aprovada")
    zip_path = root / "data/raw/uci_har_official.zip"
    saved_manifest = json.loads((root / "artifacts/data_manifest.json").read_text(encoding="utf8"))
    if sha256(zip_path) != saved_manifest["sha256"]:
        raise RuntimeError("O ZIP mudou desde a auditoria; reaudite")
    return audit_result
