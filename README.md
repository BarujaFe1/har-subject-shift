# Human Activity Recognition under Subject Shift

**Estudo de caso reprodutível — Estatística & Ciência de Dados / ICMC-USP**  
**Status de entrega:** implementação e 9 testes unitários aprovados; **dataset oficial NÃO baixado neste ambiente**; portanto **nenhum resultado experimental real ou notebook com dados UCI foi executado aqui**. Ver `reports/STATUS.md`.

## Português

### A pergunta científica

**Um reconhecimento de atividades estimado por janelas aleatórias parece melhor do que uma avaliação em pessoas nunca vistas pelo modelo?** O UCI HAR oferece **10.299 janelas**, **30 sujeitos**, **6 classes** e **561 atributos**. A unidade de predição é a janela; a unidade de independência e do bootstrap é o **sujeito**. Não é um sistema médico e não identifica pessoas.

- **H1:** comparar `GroupKFold(5)` vs `StratifiedKFold(5)` sobre as **mesmas janelas do treino oficial** com regressão logística fixa. O segundo braço é *intencionalmente inadequado* para estimar generalização a sujeitos inéditos, e nunca escolhe o modelo final.
- **H2:** seleção de 64/128/256/todos os atributos dentro de `Pipeline`; examinar desempenho e Jaccard de estabilidade, apenas em treino. É ablação exploratória, nunca tuning a posteriori pelo teste.
- **H3:** avaliar taxas de erros das atividades estáticas/dinâmicas (e confusões específicas), sem presumir o resultado.

**Teste oficial:** os nove sujeitos nunca vistos são avaliados **uma única vez**, depois de congelar todo o protocolo. A seleção de modelos usa cinco folds externos por sujeito e três internos por sujeito. Métricas: macro-F1 para as seis classes fixas, balanced accuracy, precisão/recall/F1 por classe, média e dispersão por sujeito, confusão, bootstrap por sujeito (B=2.000), latência e custo. OIC de nove sujeitos é descritivo e limitado.

### Instalação (Linux)

```bash
cd har-subject-shift
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
```

Pode usar `requirements-lock.txt` como captura do ambiente de construção, mas ambientes diferentes (Python/OS) exigem nova resolução; não prometemos transposição universal de versões binárias.

### Rodar na ordem sem vazamento

```bash
har-shift acquire --root .           # requer internet; download UCI oficial
har-shift audit --root .             # ZIP + CRC + SHA + X/y/subjects + classes
har-shift train --root . --mode full # 6 modelos, CV aninhada, H1/H2, LOCK
har-shift test --root .             # TESTE ÚNICO, bloqueia repetição
har-shift figures --root .          # F01..F14, figuras reais + manifesto
```

Para teste de infraestrutura mais barato, `--mode quick` reduz os grids para 1 configuração por família, **sem remover famílias nem a validação agrupada**. `--skip-hypotheses` é apenas diagnóstico de engenharia, não estudo completo. O comando `har-shift all --mode full` executa a sequência inteira quando online.

**Sem internet:** baixe o ZIP original por navegador da [página UCI](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) e salve com o nome `data/raw/uci_har_official.zip`; então comece em `audit`. **Não** use cópias de terceiros sem registrar a substituição de fonte.

O notebook `notebooks/analise_completa.ipynb` orquestra a mesma sequência, mostra EDA, métricas e figuras geradas, e **não reabre o teste depois de executado**. Execute com kernel limpo a partir da raiz do projeto:

```bash
jupyter nbconvert --to notebook --execute notebooks/analise_completa.ipynb \
    --output analise_completa_executada.ipynb --ExecutePreprocessor.timeout=-1
```

Se a rede não funcionar e o ZIP não estiver presente, o notebook encerra com erro explícito em G0; **não substitui dados reais por dados sintéticos**. Não apagar `artifacts/test_started.json` para refazer uma avaliação: isso destruiria o protocolo de teste único. Se houver falha após o marcador, investigar e registrar a falha antes de qualquer decisão posterior.

### Artefatos e rastreabilidade

- `docs/DOSSIER_HAR.md` e `docs/P01B_HAR_Implementacao.md`: fontes de requisitos originais recebidas.
- `data_manifest.json`, `dataset_audit.json`: hash e integridade do ZIP, números observados, contagens de classes, pessoas e features.
- `fold_subjects.json`, `oof_predictions.csv`, `cv_metrics.csv`, `subject_metrics.csv`, `model_selection.json`: seleção **sem abrir o teste**.
- `protocol_locked.json`, `test_started.json`: registro do congelamento e da avaliação única.
- `test_metrics.json`, `test_predictions.csv`, `test_subject_metrics.csv`, `best_pipeline.joblib`: resultados **apenas se houver avaliação real**.
- `h1_protocol_contrast.json`, `h2_feature_ablation.csv`: contraste metodológico/ablação de atributos, ambos internos ao treino.
- `figures/F01...F14` e `artifacts/figures_manifest.csv`: exportação PNG/SVG com proveniência.
- `reports/interview_brief.md`: respostas para entrevista técnica.

**Integridade importante:** `subject_id` e `row_id` **nunca** entram nas 561 features. Os 561 nomes de atributos podem se repetir; os IDs de coluna canônicos são `f001..f561`. Pré-processamento/seleção só é ajustado em subtreinos. `n=10.299` significa janelas, não pessoas. Um macro-F1 agregado por janelas **não** equivale à média de macro-F1 dos indivíduos; ambos são apresentados.

**Licenciamento:** página oficial UCI indica CC BY 4.0, enquanto README histórico reproduzido contém restrição comercial. Veja [LICENSE_DATA_NOTICE.md](LICENSE_DATA_NOTICE.md). Não redistribuir dados; esclarecer termos antes de uso comercial. DOI do dataset: `10.24432/C54S4K`.

**Limitações:** o aparelho era usado na cintura em atividades controladas. Não extrapolar para relógios, outro posicionamento do celular, condições clínicas ou mundo real. A comparação H1 não identifica causalmente o efeito da sobreposição temporal. Selecionar o melhor entre vários modelos via OOF produz viés de seleção (*winner's curse*), minimizado mas não eliminado pelo teste final. Há nove sujeitos no teste, logo intervalos bootstrap são instáveis.

## English — Quick start

**Research question:** Does random-window validation inflate estimates of cross-person Human Activity Recognition performance? We use UCI HAR Dataset 240 and preserve the official person-disjoint train/test split. An outer 5-fold `GroupKFold` with inner 3-fold grouped tuning selects among dummy, regularized logistic regression, kNN, linear SVM, RBF SVM, and random forest. Only after protocol locking do we evaluate the official 9-person holdout **once**. The random-window CV arm is a clearly labeled methodological illustration, never used to select models. Features are selected only inside fold-local pipelines; uncertainty is bootstrapped at the person level.

From Linux: `python3 -m venv .venv && source .venv/bin/activate && pip install -e '.[dev]'`; then `har-shift acquire`, `har-shift audit`, `har-shift train --mode full`, `har-shift test`, `har-shift figures`. Run `python -m pytest` first. See the Portuguese section for manual offline download, the notebook, artifacts and limitations. This project is **not validated for clinical use**.

### References

1. Reyes-Ortiz, Anguita, Ghio, Oneto, Parra (2013). *Human Activity Recognition Using Smartphones*. UCI, DOI: https://doi.org/10.24432/C54S4K
2. scikit-learn — GroupKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
3. scikit-learn — Common pitfalls: https://scikit-learn.org/stable/common_pitfalls.html
4. Jordao et al. (2019) — Subject Cross Validation in HAR: https://arxiv.org/abs/1904.02666
5. Historical dataset README reproduction: https://github.com/aannasw/uci-har/blob/master/data/UCI-HAR-Dataset/README.txt
