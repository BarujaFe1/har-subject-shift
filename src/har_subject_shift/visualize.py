"""Scientific figures in Portuguese. Every numerical graph uses measured artifacts."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .data import LABELS, load_split

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False, "axes.labelsize": 10, "axes.titlesize": 12, "figure.constrained_layout.use": True})


def generate(root: Path) -> list:
    ar, figdir = root / "artifacts", root / "figures"
    if not (ar / "dataset_audit.json").exists():
        raise RuntimeError("Auditoria obrigatória")
    figdir.mkdir(exist_ok=True, parents=True)
    audit = json.loads((ar / "dataset_audit.json").read_text(encoding="utf8"))
    manifest = []

    def save(fig, label, provenance):
        path = figdir / (label + ".png")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(figdir / (label + ".svg"), bbox_inches="tight")
        plt.close(fig)
        manifest.append({"figure": label, "file": str(path.relative_to(root)), "provenance": provenance})

    # F01: conceptual, no model scores
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2); ax.axis("off")
    texts = ["Celular na cintura\naceleração + giro", "Janela 2,56 s\n128 pontos × 9 sinais", "561 atributos\n(originais)", "Modelo treinado\nsem o novo sujeito", "Atividade\n1 entre 6"]
    for i, t in enumerate(texts):
        ax.text(i*2 + 1, 1, t, ha="center", va="center", bbox={"boxstyle": "round,pad=.5", "facecolor": "#edf4fb", "edgecolor": "#4b7395"}, fontsize=9)
        if i < len(texts)-1:
            ax.annotate("", xy=(i*2+1.77, 1), xytext=(i*2+1.28,1), arrowprops={"arrowstyle": "->"})
    ax.set_title("F01 — Fluxo de reconhecimento de atividade (diagrama conceitual)")
    save(fig, "F01_esquema_coleta", "diagrama_autoral")

    fig, ax = plt.subplots(figsize=(9, 3.8))
    tr_ids = audit["splits"]["train"]["subject_ids"]
    te_ids = audit["splits"]["test"]["subject_ids"]
    ax.scatter(tr_ids, np.ones(len(tr_ids)), s=110, label="Treino: 21 pessoas")
    ax.scatter(te_ids, np.zeros(len(te_ids)), s=110, label="Teste: 9 pessoas")
    ax.set(yticks=[0,1], yticklabels=["Teste", "Treino"], xticks=range(1,31), xlabel="Identificador do participante", title="F02 — Divisão oficial por pessoas (sem interseção)")
    ax.grid(axis="x", alpha=.15)
    save(fig, "F02_divisao_pessoas", "dataset_audit")

    labels = list(LABELS.values())
    train_counts = [audit["splits"]["train"]["class_counts"][name] for name in labels]
    test_counts = [audit["splits"]["test"]["class_counts"][name] for name in labels]
    fig, ax = plt.subplots(figsize=(10,4.4))
    loc = np.arange(6)
    ax.bar(loc-.2, train_counts, .4, label="Treino")
    ax.bar(loc+.2, test_counts, .4, label="Teste: descrição estrutural")
    ax.set(xticks=loc, xticklabels=labels, ylabel="Janelas", title="F03 — Distribuição das seis atividades")
    ax.tick_params(axis="x", labelrotation=18)
    ax.legend()
    save(fig, "F03_distribuicao_classes", "dataset_audit_descriptive")

    ctab = pd.read_csv(ar / "subject_activity_train.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(9,6))
    im = ax.imshow(ctab.to_numpy(), aspect="auto", interpolation="none")
    ax.set(xticks=range(6), xticklabels=list(LABELS.values()), yticks=range(len(ctab)), yticklabels=ctab.index, xlabel="Atividade", ylabel="Participante de treino", title="F04 — Janelas por participante e atividade (treino)")
    ax.tick_params(axis="x", labelrotation=25)
    fig.colorbar(im, ax=ax, label="Nº de janelas")
    save(fig, "F04_sujeito_atividade", "train_audit")

    x, _, _ = load_split(root, "train")
    # Fixed diagnostic indices, selected before inspecting test or relevance scores.
    cols = list(range(0, 561, 56))[:10]
    corr = np.corrcoef(x[:, cols], rowvar=False)
    fig, ax = plt.subplots(figsize=(6.6,5.5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set(xticks=range(len(cols)), xticklabels=[f"f{k+1:03d}" for k in cols], yticks=range(len(cols)), yticklabels=[f"f{k+1:03d}" for k in cols], title="F05 — Correlação de 10 atributos fixados (treino)")
    ax.tick_params(axis="x", labelrotation=45)
    fig.colorbar(im, ax=ax, label="Correlação de Pearson")
    save(fig, "F05_correlacoes", "train_only_features_fixed_indices")

    if (ar / "model_selection.json").exists():
        selection = json.loads((ar / "model_selection.json").read_text())
        chosen = selection["model"]
        folds = json.loads((ar / f"{chosen}_fold_metrics.json").read_text())
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar([f["fold"] for f in folds], [f["macro_f1"] for f in folds])
        ax.set(xlabel="Fold externo (sujeitos disjuntos)", ylabel="Macro-F1", ylim=(0,1), title=f"F06 — Desempenho por fold, {chosen}")
        save(fig, "F06_cv_pessoas", "train_nested_cv")

    if (ar / "h1_protocol_contrast.json").exists():
        h = json.loads((ar / "h1_protocol_contrast.json").read_text())
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(["GroupKFold\n(pessoas inéditas)", "StratifiedKFold\n(janelas aleatórias)"], [h["correct_grouped_macro_f1"], h["incorrect_random_window_macro_f1"]])
        ax.set(ylim=(0,1), ylabel="Macro-F1 no treino oficial", title="F07 — Mesmo algoritmo, perguntas diferentes")
        ax.text(.5, .08, "O segundo protocolo NÃO estima generalização a pessoas inéditas", ha="center", transform=ax.transAxes, fontsize=9)
        save(fig, "F07_contraste_protocolos", "train_H1_illustrative_only")

    if (ar / "cv_metrics.csv").exists():
        df = pd.read_csv(ar / "cv_metrics.csv").sort_values("macro_f1")
        fig, ax = plt.subplots(figsize=(8,4.5))
        ax.barh(df.model, df.macro_f1)
        ax.set(xlim=(0,1), xlabel="Macro-F1 OOF por participantes", title="F08 — Candidatos (apenas treino)")
        save(fig, "F08_candidatos", "train_nested_cv")

        sub = pd.read_csv(ar / "subject_metrics.csv")
        fig, ax = plt.subplots(figsize=(8,4.5))
        selected = sub.loc[sub.model == df.iloc[-1].model].sort_values("subject_id")
        ax.scatter(selected.subject_id, selected.macro_f1)
        ax.set(xticks=selected.subject_id.to_list(), ylim=(0,1), xlabel="Participante de treino (OOF)", ylabel="Macro-F1 individual (6 classes fixas)", title="F09 — Generalização heterogênea entre pessoas")
        save(fig, "F09_por_pessoa", "train_oof_subjects")

        runtime = pd.read_csv(ar / "runtime.csv")
        merged = df.merge(runtime, on="model")
        fig, ax = plt.subplots(figsize=(8,4.5))
        ax.scatter(merged.total_nested_cv_seconds, merged.macro_f1, s=65)
        for row in merged.itertuples():
            ax.annotate(row.model, (row.total_nested_cv_seconds, row.macro_f1), xytext=(3,3), textcoords="offset points", fontsize=8)
        ax.set(xlabel="Tempo total de validação aninhada (s)", ylabel="Macro-F1 OOF", title="F12 — Desempenho × custo computacional")
        save(fig, "F12_desempenho_custo", "train_nested_cv_runtime")

    if (ar / "h2_feature_ablation.csv").exists():
        df = pd.read_csv(ar / "h2_feature_ablation.csv")
        df = df[df.fold.astype(str)=="OOF"]
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(df.k, df.jaccard_mean, "o-", label="Jaccard entre pares de folds")
        ax.set(ylim=(0,1), xlabel="Nº de atributos selecionados", ylabel="Índice Jaccard", title="F13 — Estabilidade da seleção de atributos (treino)")
        save(fig, "F13_estabilidade_selecao", "train_H2_exploratory")

    if (ar / "test_metrics.json").exists():
        m = json.loads((ar / "test_metrics.json").read_text())
        cm = np.asarray(m["confusion"])
        fig, axes = plt.subplots(1,2,figsize=(12.5,4.8))
        for ax, data, title in zip(axes, [cm, cm/np.maximum(cm.sum(axis=1, keepdims=True),1)], ["Contagem", "Normalizada pela classe real"]):
            im = ax.imshow(data, cmap="Blues", vmin=0)
            ax.set(xticks=range(6), yticks=range(6), xticklabels=list(LABELS.values()), yticklabels=list(LABELS.values()), xlabel="Prevista", ylabel="Real", title=title)
            ax.tick_params(axis="x", labelrotation=35)
            for i in range(6):
                for j in range(6):
                    ax.text(j, i, f"{data[i,j]:.2f}" if title.startswith("Normalizada") else str(data[i,j]), ha="center", va="center", fontsize=7, color="white" if data[i,j] > .7 * data.max() else "black")
        fig.suptitle("F10 — Confusão no teste oficial: nove pessoas inéditas")
        save(fig, "F10_matriz_confusao", "test_final_one_time")

        fam = m["family_errors"]
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(["Estáticas", "Dinâmicas"], [fam["static"]["error_rate"], fam["dynamic"]["error_rate"]])
        ax.set(ylim=(0,1), ylabel="Erros / janelas verdadeiras da família", title="F11 — Taxa de erro por família de atividades")
        save(fig, "F11_erro_familia", "test_final_one_time")

        ci = m["bootstrap_subjects"]
        fig, ax = plt.subplots(figsize=(7,3.5))
        center = ci["global_macro_f1"]
        lo, hi = ci["global_macro_f1_ci_95"]
        ax.errorbar(center, 0, xerr=[[center-lo], [hi-center]], fmt="o", capsize=7)
        ax.set(ylim=(-.6,.6), yticks=[0], yticklabels=["Modelo final"], xlim=(0,1), xlabel="Macro-F1", title="F14 — IC bootstrap por pessoa (n=9; descritivo)")
        save(fig, "F14_incerteza", "test_final_bootstrap_by_subject")

    pd.DataFrame(manifest).to_csv(ar / "figures_manifest.csv", index=False)
    return manifest
