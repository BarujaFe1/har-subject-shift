# Protocolo pré-teste

**Questão:** desempenho ao prever atividades de sujeitos inéditos usando UCI HAR vs estimativa incorreta por janelas aleatórias.

## Desenho

- Unidade amostral para predição: janela de 128 amostras (2,56 s); grupo de dependência: pessoa.
- Matriz preditiva: 561 colunas de `X_train` separadas de `y_train` e `subject_train`; nunca incluir IDs.
- Dataset oficial: 7.352 janelas de 21 pessoas para desenvolvimento; 2.947 janelas de 9 pessoas para teste final. Valores esperados, não verificados aqui.
- Externo: `GroupKFold(n_splits=5)`; interno `GroupKFold(n_splits=3)`; métrica primária seis classes macro-F1; secundária balanced accuracy, erros por classe e sujeito.
- Modelos: Dummy, logística, kNN, SVM linear, SVM RBF, RF; grids de `models.py` no modo `full`.
- Critério de escolha: maior macro-F1 OOF aninhado; empate menor desvio entre sujeitos; depois ordem alfabética. Comparar famílias torna métrica selecionada sujeita a viés de seleção.
- H1: regressão logística **fixa C=1**, treino oficial em ambos braços, 5 folds por pessoa vs 5 folds aleatórios estratificados por janela. A diferença é descritiva.
- H2: k=64/128/256/all exclusivamente em folds de treino; reporte Jaccard dos índices de atributo entre folds; não selecionar modelo final via H2.
- H3: erro sobre a classe verdadeira, estáticas IDs 4–6 / dinâmicas IDs 1–3; taxas e confundibilidade.
- Bootstrap: 2.000 repetições por IDs de participante com reposição, todas as janelas do participante, CI percentil 95%, `seed=42`, com ressalva de nove sujeitos no holdout.

## Gates

G0 ZIP fonte oficial com CRC/SHA comprovados; G1 auditoria de schema e IDs aprovada; G2 EDA sem tuning no teste; G3/4 CV aninhada e OOF uma vez por linha; G5 H1/H2; G6 congelamento com SHA dos scripts e dataset; G7 avaliação única após marcador `test_started.json`; G8 figuras, discussão e documentação.

Um resultado favorável a H1/H2/H3 não é pré-condição de sucesso. O código não contém valores inventados de desempenho.
