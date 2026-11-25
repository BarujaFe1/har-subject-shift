# Reconhecimento de Atividades Humanas: generalização para pessoas não vistas

**Human Activity Recognition Under Subject Shift — Dossiê científico e plano de implementação**  
**Projeto:** P01A · Estatística e Ciência de Dados · SME0828 / ICMC-USP  
**Data de corte da pesquisa:** 21 de novembro de 2026  
**Natureza:** revisão documental e protocolo pré-registrável; **não contém treinamento, medições próprias de desempenho nem resultados simulados**.

> **Status de evidência.** **[O]** = confirmado em fonte oficial; **[R]** = observado em réplica ou literatura secundária, ainda requer auditoria do ZIP original; **[P]** = planejado/não executado; **[NV]** = não verificado diretamente. Sempre que uma quantidade se referir a *janelas*, ela **não** representa participantes independentes.

## Sumário executivo

**Decisão:** adotar **UCI Human Activity Recognition Using Smartphones (UCI HAR; UCI nº 240)** como conjunto principal **condicionado a uma auditoria local do arquivo original**, preservando sua divisão oficial por participantes. O conjunto é pequeno o bastante para computadores comuns e permite isolar a pergunta científica: *a validação por observações de pessoas já vistas apresenta uma avaliação otimista em comparação com validação por pessoas inéditas?* A página oficial documenta 10.299 janelas, 30 voluntários, seis classes, 561 atributos e 50% de sobreposição temporal [S1]. Réplicas e trabalhos publicados documentam 7.352 janelas de treino/21 participantes e 2.947 de teste/nove participantes [S4–S6].

**Recomendação metodológica:** comparar validação **por grupos de participantes** com validação ingênua por janelas **exclusivamente dentro do treino oficial**; escolher algoritmos e hiperparâmetros sem usar o teste; congelar decisões; executar o teste oficial uma única vez; reportar macro-F1, balanced accuracy, erros por atividade e dispersão entre sujeitos, com incerteza calculada no nível **participante**, não janela. A literatura aponta que sobreposição e particionamento por janelas podem inflar estimativas de desempenho; o tamanho e até o sinal da diferença **neste projeto** são desconhecidos [S7–S9].

**Reserva científica:** **PAMAP2** caso seja importante estudar sensores e atividades adicionais, aceitando muito mais trabalho de preparação e apenas nove pessoas; **HHAR** fica como extensão de robustez a novos dispositivos, mas é inadequado como primeira implementação simples em CPU. A licença atual catalogada dos três datasets é **CC BY 4.0** [S1–S3]. Há, porém, **uma divergência real**: cópias do `README.txt` original do UCI HAR proíbem uso comercial [S4, S13]. Para uso comercial ou redistribuição em produto, solicitar esclarecimento aos detentores dos direitos; não considerar essa divergência resolvida pela pesquisa.

**Limitação da auditoria agora:** a página da UCI e o endereço oficial de download foram alcançados via navegador, mas a transferência e descompactação integral do ZIP de 58,2 MB **não foram concluídas neste ambiente** (limitação do mecanismo de leitura de arquivos grandes e ausência de rede no executor local). Os caminhos, IDs de sujeitos e frequências abaixo foram triangulados por documentação e réplicas, **não certificados por leitura de cada byte do ZIP original**. A auditoria local antes de implementar é um **gate obrigatório**.

---

## 1. Problema, população e estimando

### 1.1 Pergunta principal

Quanto muda a qualidade de reconhecimento das seis atividades quando um modelo treinado com pessoas A é avaliado em pessoas B **inteiramente ausentes do treinamento**, relativamente a um procedimento ingênuo que separa janelas aleatoriamente e admite a mesma pessoa nos dois subconjuntos?

É essencial distinguir três tarefas:

| Tarefa | Entrada | Saída desejada | O que se aprende / testa |
|---|---|---|---|
| **Reconhecimento de atividade (HAR)** | Estatísticas dos sinais inerciais de uma janela | Uma das seis atividades | Padrões de movimento/postura |
| **Identificação de pessoa** | Movimento de uma pessoa | Identidade do portador | Assinaturas individuais, **não** o objetivo deste trabalho |
| **HAR generalizável entre pessoas** | Mesmas estatísticas de janela | Atividade em indivíduos não vistos | Robustez à mudança de pessoa (*subject shift*) |

**Unidade de predição:** janela de 2,56 segundos, com 128 leituras a 50 Hz. **Unidade de independência para separação e inferência:** participante identificado por `subject_id`. **População acessível:** adultos voluntários de 19–48 anos, em atividades roteirizadas, com um Samsung Galaxy S II na cintura [S1]. **Alvo de generalização efetivamente testável:** outras pessoas desse mesmo contexto experimental; **não** toda a população, idosos frágeis, pacientes, smartwatches ou telefones em bolsos.

**Estimando primário de desempenho:** macro-F1 de previsões *out-of-fold* sob validação por pessoa no treino e macro-F1 do teste oficial de pessoas inéditas, com a natureza de cada estimativa explicitada. O contraste `Δ = macroF1_linhas − macroF1_grupos` é **diagnóstico do protocolo de avaliação**, não um efeito causal universal do *leakage* nem uma estimativa de desempenho clínico.

### 1.2 Relevância aplicada e limites

Acelerômetros medem aceleração em três eixos; giroscópios, velocidade angular. Em smartphones e relógios, esses sensores sustentam reconhecimento de marcha, atividades cotidianas, treinamento físico e pesquisa em saúde digital [S1, S10, S11]. O UCI HAR disponibiliza vetores de atributos construídos de sinais inerciais, e também séries inerciais segmentadas para quem desejar explorá-las posteriormente [S4]. **Não mede** diagnóstico, quedas reais, progressão de doença, adesão clínica nem confiabilidade de dispositivos médicos; qualquer conclusão clínica seria extrapolação não autorizada.

O trabalho fundador de **Anguita et al. (2013)** apresenta o benchmark de 30 pessoas e um classificador SVM multiclasse [S5]. **Stisen et al. (2015)** documentam que heterogeneidade de modelos e sensores prejudica a transportabilidade de HAR [S10]. Pesquisas metodológicas de 2019 mostram por que janelas temporalmente próximas e participantes repetidos exigem validação adequada [S7, S8]. Estudos recentes ainda distinguem desempenho em protocolo de laboratório e transferência a outros sujeitos [S11, S12]. **Nenhum resultado publicado será adotado como resultado deste projeto.**

## 2. Comparação e auditoria dos datasets

### 2.1 Quadro de escolha

| Dataset | Escala catalogada | Pessoas/classes | Unidade original e sensores | Tamanho informado pela UCI | Partição pronta por participante | Licença da página oficial | Adequação |
|---|---|---|---|---|---|---|---|
| **UCI HAR [S1]** | **10.299 janelas**, 561 atributos calculados | **30 / 6** | Janelas 2,56 s, 50 Hz; acelerômetro + giroscópio de smartphone na cintura | **58,2 MB (ZIP)** | **Sim**, 70%/30% dos voluntários; ID em arquivos separados | CC BY 4.0 **com ressalva de conflito com README** | **Primário**: alinhamento excelente à pergunta, operação leve |
| **HHAR [S2]** | **43.930.257 registros** catalogados; **não são janelas prontas comparáveis ao UCI HAR** | **9 usuários / 6 atividades** no experimento HAR | Amostras de acelerômetro/giroscópio, oito smartphones e quatro smartwatches, frequências por dispositivo; dados de experimento estático adicionais | `Activity recognition exp.zip` **741,1 MB**; `Still exp.zip` **42,9 MB**; total anunciado **784 MB** | **Não documentada como partição oficial pronta**; há coluna `User`, exige construção de janelas e divisão por usuário | CC BY 4.0 | Extensão para **heterogeneidade de aparelho**; alto custo de limpeza |
| **PAMAP2 [S3]** | **3.850.505 linhas** catalogadas; **não janelas prontas** | **9 / 18 atividades possíveis**; protocolo principal com 12 | Sinais brutos a 100 Hz, três IMUs (punho, peito, tornozelo) e medidor cardíaco ~9 Hz | **656,3 MB (ZIP)** | **Não há split oficial pronto** documentado; arquivos por sujeito/sessão permitem reconstruí-lo | CC BY 4.0 | **Reserva** para estudo com mais modalidades, sob maior custo de ETL |

**Cuidado com denominadores:** comparar 43,9 milhões de linhas do HHAR com 10.299 janelas do UCI HAR como se fossem exemplos equivalentes é incorreto. Os dois primeiros são escalas de granularidade distintas. Em HHAR e PAMAP2 a construção de janelas, a agregação e a gestão de sinais ausentes precisam ser definidas antes de classificar [S2, S3].

### 2.2 Downloads, licença e rastreabilidade

- **UCI HAR:** página oficial `https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones`, DOI `10.24432/C54S4K`. O botão oficial apontou para `https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip` [O: link resolvido, **NV: arquivo inteiro baixado**]. A UCI anuncia **58,2 MB comprimidos** [S1].
- **HHAR:** `https://archive.ics.uci.edu/dataset/344/hhar`, DOI `10.24432/C5689X`. O botão oficial apontou para `https://archive.ics.uci.edu/static/public/344/heterogeneity%2Bactivity%2Brecognition.zip` [O: link, NV: bytes extraídos] [S2].
- **PAMAP2:** `https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring`, DOI `10.24432/C5NW2H`. O botão oficial apontou para `https://archive.ics.uci.edu/static/public/231/pamap2%2Bphysical%2Bactivity%2Bmonitoring.zip` [O: link, NV: bytes extraídos] [S3].
- **Licenças:** as **páginas UCI atuais** identificam os três conjuntos como CC BY 4.0, com atribuição (fonte, autores, licença e indicação de modificações). **Conflito específico do UCI HAR:** README histórico afirma literalmente *“Any commercial use is prohibited.”* [S4, S13], em aparente tensão com o catálogo. **Não resolver juridicamente por inferência**: usar academicamente com citação; antes de exploração comercial, obter confirmação dos titulares/instituição.
- **Tamanho após descompactação:** **NV** neste ambiente. Não apresentar o tamanho comprimido como espaço ocupado depois de extrair; na implementação, registrar `du -sh` do diretório extraído, bytes do ZIP, SHA-256, data e URL de origem.
- **Disponibilidade de download real:** links do servidor oficial são observáveis e a página é pública, mas a ferramenta de navegação recusou ler o ZIP grande, e o executor de código não tinha conectividade DNS. Assim, **não afirmamos ter baixado o ZIP**. Não houve falha do dataset demonstrada.

### 2.3 Decisão de dataset

**Primário: UCI HAR**, pela existência de `subject_train.txt`/`subject_test.txt`, atributos estruturados, seis classes observáveis, tamanho reduzido, partida oficial por pessoa e compatibilidade com regressão logística, k-NN, SVM e florestas aleatórias [S1, S4]. **Reserva: PAMAP2**, se o principal ficar indisponível ou se a ênfase migrar para sinais brutos e múltiplos sensores; exigirá desenhar janelas por sessão e validação por pessoa, tratando NaNs. **HHAR como segundo estudo**, não reserva operacional imediata: seu objetivo distintivo é a generalização **entre dispositivos** e usuários; misturar *subject shift* e *device shift* no primeiro artigo dificultaria interpretação [S2, S10].

**Regra de bloqueio:** só congelar definitivamente o conjunto primário após verificar localmente: integridade do ZIP, árvores de pastas, shapes, domínios de `subject_id`, contagens por classe e participantes, ausência de interseção treino/teste, licença dentro do pacote e checksums.

## 3. Estrutura documental do UCI HAR e dicionário de dados

### 3.1 Caminhos esperados — estrutura descrita no README reproduzido [R, S4]

```text
UCI HAR Dataset/
├── README.txt
├── features_info.txt
├── features.txt
├── activity_labels.txt
├── train/
│   ├── X_train.txt
│   ├── y_train.txt
│   ├── subject_train.txt
│   └── Inertial Signals/
│       ├── body_acc_x_train.txt, body_acc_y_train.txt, body_acc_z_train.txt
│       ├── body_gyro_x_train.txt, body_gyro_y_train.txt, body_gyro_z_train.txt
│       └── total_acc_x_train.txt, total_acc_y_train.txt, total_acc_z_train.txt
└── test/
    ├── X_test.txt
    ├── y_test.txt
    ├── subject_test.txt
    └── Inertial Signals/ (nove canais equivalentes com sufixo _test)
```

**Verificação ainda pendente:** confrontar `zipfile.namelist()` com essa árvore; não presumir que uma cópia de terceiros seja idêntica em conteúdo e licença ao pacote da UCI.

### 3.2 Dicionário de dados operacional

| Campo/arquivo | Granularidade | Tipo, domínio e uso | Regras de auditoria |
|---|---|---|---|
| `X_train.txt` | Uma linha por janela do treino | Matriz numérica **7.352 × 561** [R]; atributos de tempo/frequência, normalizados no pacote de origem | Somente valores finitos; 561 colunas; linhas correspondentes a `y_train` e `subject_train` |
| `X_test.txt` | Uma linha por janela do teste | Matriz **2.947 × 561** [R]; **teste final intocado** | Mesmo dicionário, mesma ordem de 561 posições, nunca usar em `fit`/seleção |
| `y_train.txt`, `y_test.txt` | Um código por janela | `activity_id ∈ {1,2,3,4,5,6}` | Sem nulos e alinhados com `X_*` |
| `subject_train.txt`, `subject_test.txt` | Um ID por janela | Inteiro de 1 a 30; **grupo**, não atributo preditivo | Conjuntos disjuntos treino/teste; não introduzir `subject_id` em `X` |
| `activity_labels.txt` | Uma linha por classe | Mapa **1 WALKING; 2 WALKING_UPSTAIRS; 3 WALKING_DOWNSTAIRS; 4 SITTING; 5 STANDING; 6 LAYING** | Classe consistente com arquivo de destino; ordem estável nos gráficos |
| `features.txt` | Uma entrada por posição do vetor | Índice 1–561 + nome humano (ex.: `tBodyAcc-mean()-X`, `fBodyAcc-std()-Y`) | **Nomes podem se repetir**: manter índice como identificador canônico (`f001`…`f561`) e nome original como metadado [S14] |
| `features_info.txt` | Dicionário, não matriz | Definições de médias, desvios, energia, FFT e outras funções | Não converter a descrição em dado de treino |
| `Inertial Signals/*` | Janela × 128 amostras por eixo | Nove sinais: body_acc XYZ, body_gyro XYZ, total_acc XYZ | **Opcionais**; alinhamento por linha; não exigidos para análise principal de 561 atributos |
| `split` | Metadado operacional | `train` ou `test` | Nunca misturar para ajuste/validação |
| `row_id` | Metadado derivado | Índice estável dentro do split, ex.: `train_000001` | Rastrear previsões; **não** usar como feature |

**Observação sobre normalização original:** o README reproduzido informa atributos numéricos limitados a `[-1,1]` [S4]. Isso **não autoriza** aprender reescalonamento a partir de todo o conjunto reunido. Toda `StandardScaler` adicional deve ser ajustada *dentro* das partições de treinamento de cada fold [S15].

### 3.3 Partição de participantes e contagens documentadas

| Corte | Janelas | Pessoas | Identificadores [R] |
|---|---:|---:|---|
| **Treino oficial** | **7.352** | **21** | `1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30` |
| **Teste oficial** | **2.947** | **9** | `2, 4, 9, 10, 12, 13, 18, 20, 24` |
| **Total** | **10.299** | **30** | IDs 1–30 |

Os IDs foram publicados em análise independente de repositório [S6], e a divisão 21/9 também aparece em artigo recente com separação por pessoa [S11]. **A interseção vazia é uma proposição documental, ainda dependente de conferência nos arquivos oficiais locais.**

| Atividade | Treino [R, S6] | Teste (total − treino) [R] | Total [R, S16] | Grupo analítico |
|---|---:|---:|---:|---|
| `WALKING` | 1.226 | 496 | 1.722 | Dinâmica |
| `WALKING_UPSTAIRS` | 1.073 | 471 | 1.544 | Dinâmica |
| `WALKING_DOWNSTAIRS` | 986 | 420 | 1.406 | Dinâmica |
| `SITTING` | 1.286 | 491 | 1.777 | Estática |
| `STANDING` | 1.374 | 532 | 1.906 | Estática |
| `LAYING` | 1.407 | 537 | 1.944 | Estática |
| **Total** | **7.352** | **2.947** | **10.299** | — |

**Proveniência desta tabela:** totais por classe reproduzidos em artigo publicado [S16]; coluna treino de relatório técnico de terceiro [S6]; coluna teste obtida por subtração, **não** de execução própria sobre `y_test.txt`. Confirmação de cada célula, incluindo a distribuição por **atividade × participante**, é tarefa obrigatória da auditoria local. Não interpretar essas contagens como resultados de modelagem.

### 3.4 Inspeção de consistência antes do primeiro ajuste

1. Validar arquivo compactado: tamanho, SHA-256, lista de entradas, erros de CRC, leitura segura sem extrair arquivos fora da pasta (prevenir *zip slip*).
2. Carregar `features.txt` como lista **ordenada** de 561 posições; registrar nomes repetidos, mas **não descartar colunas só porque têm nomes iguais**. Usar `f001`–`f561` evita que um parser renomeie silenciosamente colunas; duas posições com mesmo rótulo podem ter valores diferentes.
3. Confirmar `X_train.shape=(7352,561)`, `X_test.shape=(2947,561)`, dimensões de `y`, `subject`, ausência de NaN/Inf, classes conhecidas, consistência dos tipos e alinhamento linha a linha.
4. Testar `set(subject_train) ∩ set(subject_test) == ∅` e cardinalidade 21/9, registrando resultado no log. Testar também duplicatas exatas de vetores entre cortes (não equivalem necessariamente a mesmo sujeito, mas exigem diagnóstico se existirem).
5. Computar tabela `subject_id × activity_id` **apenas para diagnóstico**; todas as pessoas contêm todas as classes? qual o mínimo de janelas por sujeito/classe? Revisar a adequação dos folds a partir dessa tabela.
6. Detectar atributos constantes/variância nula **no conjunto usado para cada ajuste**, não após juntar treino e teste. Outliers/colinearidade: reportar sem eliminação arbitrária. Nunca transformar `subject_id` em feature.
7. Separar inspeção descritiva **pública e estrutural** do conjunto final da investigação: a conferência de shapes, rótulos e quantidade de participantes no teste é permitida como auditoria; **gráficos de desempenho, seleção e qualquer decisão sobre o modelo não podem ser dirigidos pelo teste**.

## 4. Hipóteses e protocolos refutáveis

### H1 — Otimismo da divisão aleatória por janela

**Hipótese prévia:** mantendo o algoritmo e o conjunto de treino oficial, a avaliação aleatória por janelas apresentará maior macro-F1 do que a validação com participantes excluídos. **Contraste descritivo:** `Δ_F1 = F1_médio_ou_OOF_linhas − F1_OOF_grupos` (pontos percentuais), e analogamente `Δ_BA`.

- **Grupo correto:** estimativas OOF em cinco folds `GroupKFold`, com sujeitos não compartilhados entre ajuste e validação.
- **Controle ingênuo/ilustrativo:** validação `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` sobre **as mesmas 7.352 janelas do treino**, mantendo classes aproximadamente equilibradas, **sem** agrupar por pessoa. Esse braço é explicitamente rotulado **“ilustração de avaliação inadequada para generalização a novos participantes”**.
- **Para isolar o protocolo:** escolher **configuração fixa antes** do experimento ilustrativo, preferencialmente o classificador linear inicial com hiperparâmetro predeterminado; executar os mesmos transformadores dentro dos respectivos folds, evitando a confusão extra de grids diferentes.
- **Refutação:** se `Δ ≤ 0`, a previsão direcional não se confirma neste conjunto/protocolo. Mesmo se `Δ > 0`, o experimento não prova que todo o excesso decorre exclusivamente de janelas sobrepostas: pessoa, sessão e proximidade temporal estão entrelaçadas.
- **Incerteza:** opcionalmente aplicar bootstrap por participante a previsões OOF emparelhadas; intervalo é **descritivo**, pois folds compartilham modelos treinados em amostras parcialmente sobrepostas. Não fabricar *p*-values de janelas independentes.

### H2 — Regularização/seleção de atributos versus desempenho, custo e estabilidade

Comparar regressão logística regularizada sem seleção, regressão logística com seleção `SelectKBest(f_classif, k∈{64,128,256})`, e uma segunda alternativa regularizada selecionada em validação interna; **todas as escolhas dentro de `Pipeline`**. Conservar o mesmo esquema de grupos. Reportar **macro-F1 por pessoa**, desvio entre pessoas, número de atributos, segundos de ajuste, MiB de modelo serializado e latência de inferência. **Refutação:** se redução de atributos não melhorar custo mensurável ou piorar substancialmente a generalização/estabilidade, não anunciar benefício. Comparar também contra `k=all` sem manipular o teste.

A seleção univariada foi escolhida por simplicidade curricular, **não** por presumir independência dos atributos: correlações podem tornar nomes/valores redundantes e comprometer interpretações isoladas [S14, S17]. Métrica de estabilidade de seleção: **Jaccard médio** dos conjuntos de índices de features escolhidos entre folds, sempre referindo-se ao mesmo `k`.

### H3 — Confusões diferentes em atividades estáticas/dinâmicas

Definir de antemão `STATIC = {SITTING, STANDING, LAYING}` e `DYNAMIC = {WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS}`. Para cada família, calcular `(erros de janelas verdadeiras da família)/(total de janelas verdadeiras da família)`, e separar **erros dentro da família** de **erros cruzando famílias**. Analisar confusão específica `SITTING↔STANDING` e `WALKING_UPSTAIRS↔WALKING_DOWNSTAIRS`, sem assumir qual será maior. **Refutação:** diferenças nulas, invertidas ou sem suporte da variabilidade entre sujeitos rejeitam uma afirmação geral de maior dificuldade estática.

**Evitar interpretação enganosa:** não comparar só números absolutos de erros quando os denominadores por classe são diferentes; acrescentar recalls por classe, matrizes normalizadas por classe verdadeira e intervalos por participante. A comparação H3 é **exploratória** se múltiplas confusões forem examinadas sem correção para multiplicidade.

## 5. Modelos e plano de computação em CPU

**Dados de entrada:** matriz tabular de 561 atributos, **sem** IDs ou séries cruas na primeira versão. Cada candidato vai de dados de treino a predição em `Pipeline` quando há transformações. Configurações abaixo são um **plano** — não representam busca concluída.

| Modelo | Fluxo obrigatório | Grid interno curto (exemplo pré-registrável) | Justificativa / custo |
|---|---|---|---|
| **`DummyClassifier(strategy='most_frequent')`** | Sem escala/seleção | Sem tuning | Piso interpretável; desempenho macro-F1 provavelmente baixo, **não calcular antecipadamente** |
| **Regressão logística multinomial** | `VarianceThreshold(0) → StandardScaler → LogisticRegression(solver='lbfgs', max_iter=1500)` | `C = {0.1, 1, 10}` | Baseline forte, linear, regularizado; monitorar convergência. No scikit-learn recente, `lbfgs` com seis classes faz otimização multinomial sem forçar parâmetro deprecado `multi_class` |
| **k-NN** | `VarianceThreshold → StandardScaler → KNeighborsClassifier` | `n_neighbors = {3,7,15}`; pesos uniformes no primeiro ciclo | Distâncias dependem da escala; memória de armazenamento e inferência mais onerosas |
| **SVM linear** | `VarianceThreshold → StandardScaler → LinearSVC` | `C = {0.1,1,10}` | Baseline de margem escalável; verificar convergência |
| **SVM RBF** | `VarianceThreshold → StandardScaler → SVC(kernel='rbf')` | `C = {1,10}` × `gamma = {'scale',0.01}` | Não linear; CPU mais caro, aplicar limite de tempo |
| **Random Forest** | `VarianceThreshold → RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=42)` | `min_samples_leaf = {1,3}`; `max_features='sqrt'` | Não exige escala; registra custo e importância com cautela |

**Seleção de atributos:** testar em ablação de regressão logística (`k∈{64,128,256,'all'}`) **somente dentro dos folds de treinamento**. Se a ablação for candidata à escolha final, incorporá-la à validação interna, e não escolher `k` olhando OOF do fold externo. Não adicionar busca extensa de redes neurais. Um MLP pequeno ou uma CNN 1D em CPU **pode** ser citado como trabalho futuro, nunca requisito de conclusão.

### 5.1 Esquema de validação sem contaminação

1. **Reservar e não ler para modelagem** `test/X_test.txt`, `test/y_test.txt` e `test/subject_test.txt` até o protocolo e o código estarem congelados. Registrar também o hash do ZIP.
2. **Dentro de `train/` (21 sujeitos):** cinco folds externos `GroupKFold(n_splits=5)`; em cada fold, usar **somente os sujeitos de treino externos** em `GroupKFold(n_splits=3)` para escolher hiperparâmetros. `StratifiedGroupKFold` é alternativa **somente se** a tabela sujeito×classe indicar instabilidade grave da composição dos folds; preservar não interseção, justificar a escolha e fixar a semente conforme a API [S18].
3. Em cada fold externo: fazer seleção interna; reajustar o `Pipeline` vencedor no treino externo; prever **apenas** as pessoas excluídas; guardar `subject_id`, `activity_id`, `y_pred`, `fold_id`, tempo e configurações. Ao juntar cinco folds, cada janela de treino terá **uma previsão OOF**.
4. Reportar **F1 e balanced accuracy OOF agrupados**, além da **distribuição de métricas por fold** e das métricas por pessoa. A pontuação do modelo vencedor nesses folds serve de avaliação interna; para comparar vários vencedores pós-seleção, reconhecer *winner's curse*. O **teste externo oficial** é a estimativa separada da aplicação final.
5. Selecionar o algoritmo por **média de macro-F1 nos folds internos/externos com regra definida**: maximizar macro-F1 OOF, desempatar por menor latência e menor desvio entre sujeitos; registrar escolha/justificativa **antes de abrir o teste**. Para rigor maior, distinguir os scores de **seleção** e os scores OOF de **validação aninhada**, que executam papéis diferentes.
6. Ajustar configuração final sobre **todo o treino oficial** (hiperparâmetros obtidos por `GroupKFold` interno de 3 ou 5 folds em todo o treino, sem consultar o teste). Avaliar **uma vez** em `test/` oficial, com os nove sujeitos não vistos. Não reabrir o teste para refazer tuning, refinar figuras de hiperparâmetros ou trocar vencedor.

**Por que `GroupKFold`?** A documentação oficial do scikit-learn recomenda separar amostras por grupos quando várias observações vêm da mesma pessoa, evitando estimativas otimistas de capacidade de prever novos indivíduos [S18]. **Por que `Pipeline`?** Filtro de variância, escala e seleção são aprendidos em cada subconjunto de ajuste, não vazam estatísticas do fold de validação ou do teste [S15].

### 5.2 Custo, ambiente e travas

- **Alvo operacional [P]:** Python 3.11+, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn` opcional, `joblib`, `pytest`, `psutil`; executável **em CPU, sem GPU**, em computador com **8 GiB de RAM e até dois workers**. Registrar versões exatas por `requirements-lock.txt` ou ambiente exportado. Ter oito GiB é uma **meta**, não medição da máquina do autor.
- **Tamanho de matrizes (estimativa, não tamanho ZIP):** 10.299 × 561 × 4 bytes ≈ **22,0 MiB** para `float32` (≈44,1 MiB para `float64`), **sem** contar cópias, DataFrames, vetores, pipelines nem arrays temporários. Ler `float32` quando aceito pelos algoritmos e avaliar custo de cópias internas.
- **Busca limitada [P]:** cinco folds externos × três internos; **até 15 configurações no total** para os cinco algoritmos ajustáveis acima (3+3+3+4+2) = **até 225 ajustes internos**, além de reajustes externos e seleção final; `Dummy` sem busca. `n_jobs=2` somente onde houver suporte; evitar paralelismo aninhado (`GridSearchCV(n_jobs=1)` com floresta `n_jobs=2`, ou o inverso). Não disparar grids simultâneos.
- **Orçamento de engenharia [P]:** objetivo ≤ **90 minutos de tempo de parede** na máquina de referência CPU, ≤ **6 GiB de pico** para o processo; se exceder, interromper graciosamente e reduzir **número de candidatos** ou usar seleção em fases **sem violar o desenho de grupos**. Não alegar que os limites já foram cumpridos.
- **Semente:** `random_state=42` nas fontes aleatórias (modelos, embaralhamento ilustrativo, bootstrap). `GroupKFold` sem embaralhamento por padrão; salvar lista exata de sujeitos por fold. Limitar bibliotecas BLAS a 1–2 threads, se necessário.
- **Inferência:** após `warm-up`, medir latência de uma janela e lotes de 100/1.000 janelas com `time.perf_counter()`, ≥30 repetições, reportando mediana e p95; mesmos dados e hardware, tempo total e configuração de threads documentados. Guardar tamanho `joblib` de cada modelo; não chamar milissegundos por **janela** de monitoramento clínico.

## 6. Métricas, agregação e intervalos de confiança

| Medida | Definição operacional | Unidade e interpretação |
|---|---|---|
| **Macro-F1** (principal) | Média aritmética do F1 das **seis classes fixadas**; `zero_division=0` com registro de classes ausentes | [0,1], equilibra classes sem ponderar por frequência |
| **Balanced accuracy** | Média dos recalls por classe verdadeira | [0,1], identifica desequilíbrio entre classes |
| **Matriz de confusão** | `confusion_matrix(..., labels=[1,2,3,4,5,6])`, absoluta e normalizada por linha | Número de janelas/percentual dentro de cada atividade real |
| **Por classe** | Precisão, recall, F1, suporte e exemplos de confusão | Evita esconder uma atividade ruim no macro-F1 |
| **Por participante** | Macro-F1 calculado separadamente em cada `subject_id`; média, DP, mínimo, máximo e boxplot das pessoas | Visibiliza pessoas de pior generalização; observar classes ausentes |
| **Desempenho por família** | Taxa de erro condicionada a estáticas/dinâmicas e matriz inter-/intrafamília | Compara H3 com denominadores coerentes |
| **Complexidade** | `fit_seconds`, `predict_ms_per_window`, `batch_ms`, `model_size_mib`, `n_selected_features` | Trade-off de custo e generalização |
| **Estabilidade** | DP e IQR das métricas por sujeito; Jaccard das features selecionadas por fold | Robustez descritiva entre pessoas/folds |

**Não fundir silenciosamente estimandos:** (a) calcular macro-F1 **agregando todas as janelas** de teste dá maior peso a pessoas com mais janelas; (b) calcular a **média dos macro-F1 individuais** dá igual peso a pessoas. Reportar **ambos**, explicando que não são intercambiáveis. Em folds, preferir painel de métricas por pessoa e OOF global a uma simples média não ponderada de scores de folds sem interpretação.

### Bootstrap agrupado por `subject_id` [P]

- **Teste final:** `B=2.000` réplicas bootstrap do conjunto dos **nove participantes** de teste, cada réplica sorteando **nove IDs com reposição** e incluindo todas as suas janelas/valores preditos. Repetições de um ID contam com multiplicidade. Calcular macro-F1 agregado e média por pessoa em cada réplica, IC percentil 2,5%–97,5%; registrar número de réplicas válidas e classes ausentes. **Este IC é condicional aos modelos treinados**, não inclui incerteza de coleta do dataset ou de todo o procedimento de escolha.
- **Validação OOF:** procedimento análogo com os **21 participantes** de treino e suas previsões de folds externos. O bootstrap não substitui validação por grupos e sua interpretação deve reconhecer dependência entre previsões obtidas de modelos treinados em folds comuns.
- **Comparação de dois classificadores no mesmo teste:** reamostrar os **mesmos IDs nas mesmas réplicas** (bootstrap emparelhado), reportar distribuição da diferença em pontos percentuais. Não aplicar bootstrap de janelas individuais; isso fabrica tamanho amostral efetivo.
- **Limitação:** com apenas **nove** pessoas no teste, um IC por bootstrap de participantes pode ser instável; não apresentar precisão elevada, nem interpretá-lo como garantia de validade fora do benchmark [S19].

## 7. Experimento de viés por janelas: separação visual e interpretativa

O braço ingênuo de H1 não deve contaminar a escolha final. O artigo deve ter quadros distintos:

**Estimativa A — generalização honesta:** OOF por sujeitos dentro de `train/` e avaliação **separada** na partição oficial de pessoas inéditas. **Estimativa B — demonstração de risco metodológico:** OOF por janelas **somente do treino** com `StratifiedKFold`. Se a Estimativa B for maior, isso mostra que o protocolo mede uma tarefa diferente, mais fácil/contaminada por informação intraindivíduo — **não prova** sozinho que a diferença é devida exclusivamente aos 50% de sobreposição. Se for menor, registrar e investigar em vez de descartar.

**Recomendação editorial:** título do gráfico central: **“Mesmo algoritmo, perguntas diferentes: janelas aleatórias versus pessoas inéditas”**; legenda deve indicar `n=7.352 janelas`, `21 pessoas`, protocolos internos e **não** usar o teste oficial como contraponto do braço ingênuo, porque isso misturaria diferenças de amostragem/partição. Literatura de Jordao et al. (2019) relata inflação para outros dados/procedimentos; não importar os percentuais publicados como previsão numérica aqui [S7, S8].

## 8. Visualizações planejadas (gráfico por gráfico)

| Figura | Conteúdo/variável, fonte | Como interpretar e restrição |
|---|---|---|
| **F01 — Esquema de coleta** | Smartphone na cintura, aceleração/rotação → janela 128×9 → vetor 561 → classe; **diagrama autoral** | Não sugerir continuidade temporal entre janelas de participantes distintos |
| **F02 — Quem fica fora?** | Fluxo com 30 IDs de sujeitos, 21 treino/9 teste, e cinco grupos externos do treino | Evidenciar ausência de interseções, não apenas 70/30 de linhas |
| **F03 — Distribuição de classes** | Barras lado a lado `train` vs `test`, seis atividades, contagens e percentuais (auditoria descritiva) | Conferir contra `y_*.txt`; não ajustar modelo após ver diferenças de teste |
| **F04 — Matriz sujeito×atividade** | Heatmap de contagens do **treino oficial** por 21 IDs × seis classes | Mostrar pessoas/classes escassas e qualidade dos folds |
| **F05 — Variabilidade de features** | Distribuição de variâncias e correlações amostradas no **treino** | Explicar colinearidade, constantes, possíveis nomes repetidos; sem selecionar a partir do teste |
| **F06 — Validação por pessoa** | Cinco folds com IDs de treino/validação, e F1 por fold; **apenas treino** | Demonstrar separação real entre participantes |
| **F07 — Resultado central H1** | Comparação lado a lado de macro-F1 e balanced accuracy: CV por janelas × CV por participantes no **mesmo treino** | Rótulo “avaliação ilustrativa ingênua” no braço por janelas; quantificar Δ sem generalizá-lo |
| **F08 — Modelos candidatos** | Dot-and-whisker de macro-F1 **OOF agrupado** dos candidatos, com dispersão por sujeito | Sem ranquear a partir de teste oficial; IC por pessoa quando defensável |
| **F09 — Desempenho por sujeito** | Dot/boxplot e tabela de F1/recall para IDs de treino OOF; painel **separado** para teste final | Evitar revelar identidade; IDs pseudonimizados no artigo, se possível |
| **F10 — Matriz de confusão final** | 6×6 absoluta + normalizada por verdadeira, **teste final uma única vez** | Distinguir SIT/STAND e escadas; exibir suporte por classe |
| **F11 — Erros estáticos vs dinâmicos** | Barras de erros por família, dentro e fora da família, e recalls por atividade | Denominadores por família e IC sujeito se adequado; não confirmar H3 por aparência |
| **F12 — Trade-off H2** | Dispersão: macro-F1 OOF por pessoas × latência; tamanho do ponto = nº features ou MiB de modelo | Apresentar fronteira desempenho–custo, sem dizer “ótimo” sem critério |
| **F13 — Estabilidade de seleção** | Frequência de atributos selecionados por fold e Jaccard por pares de folds, **somente treino** | Features com nomes repetidos identificadas pelo índice ordinal |
| **F14 — Incerteza do teste** | Pontos e IC 95% bootstrap de sujeitos para modelo final e baseline no teste oficial | Indicar somente **9 clusters**, limites do bootstrap e ausência de garantia clínica |

**Produção de gráficos:** exportar `.png` a 200 dpi e, quando possível, `.svg`; legendas em português, unidades e denominadores; paleta acessível; registrar geração em `figures_manifest.csv` com fonte de cada gráfico (`train_OOF`, `train_audit`, `test_final`) e versão do modelo. Nenhuma figura numérica é gerada nesta etapa de pesquisa.

## 9. Discussão de riscos e validade

| Risco | Gravidade | Mitigação exigida |
|---|---|---|
| **Mesmo sujeito nos dois lados** | Crítico | Teste automatizado de conjuntos de IDs disjuntos em cada fold e no split oficial |
| **Sobreposição temporal de 50% das janelas** | Alta | Agrupar por pessoa; se comparar janelas, rotular como experimento de viés; não afirmar independência |
| **Seleção/normalização antes do split** | Crítico | `Pipeline` e ajuste exclusivo em treino interno; teste imutável |
| **Escolher melhor modelo olhando o teste** | Crítico | Protocolo/hiperparâmetros congelados antes do `test`; relatório final com execução única |
| **Interpretar 10.299 como 10.299 pessoas** | Crítico | N de clusters = 30; IC por sujeitos, teste com nove pessoas |
| **Frequência desigual de atividades/pessoas** | Média | Macro-F1, balanced accuracy, análise por pessoa e classe, suporte explícito |
| **Sujeitos × aparelho/contexto confundidos** | Alta | Mesmo modelo de telefone na cintura, laboratório, 19–48 anos; não fazer alegação universal |
| **Nomes de features duplicados** | Média | Identificador ordinal de 561 colunas, metadados separados, não colapsar valores |
| **Dados originalmente normalizados** | Média | Conferir no README; `StandardScaler` somente em folds de treino; não misturar cortes |
| **Tuning excessivo/comparações múltiplas** | Alta | Grids curtos, nested CV, candidatos congelados, reconhecer seleção pós-hoc |
| **Apenas nove sujeitos no teste externo** | Alta | Intervalos amplos e interpretação restrita; dar distribuição individual e limites da extrapolação |
| **Conflito de licença README × catálogo** | Alta para uso comercial | Documentar ambos, citar fonte/termos e solicitar orientação antes de exploração comercial |
| **Falha de download/ZIP incorreto** | Alta operacional | Obter diretamente UCI, verificar checksums, contagem, estrutura, registro de bytes e provenance |
| **Reidentificação de padrões de movimento** | Média | Não usar `subject_id` como feature nem divulgar sinais associáveis a pessoa; manter apenas IDs mínimos necessários |

**Validade externa:** UCI HAR não representa relógios no punho, celulares no bolso, uso cotidiano não roteirizado, crianças, idosos, usuários com limitações motoras ou mudanças de aparelhos. O deslocamento entre **sujeitos** é apenas uma dimensão do deslocamento de domínio; **HHAR** é especialmente pertinente para estudar hardware, e **PAMAP2** para sensores e atividades adicionais [S2, S3, S10, S12].

**Nota científica:** o desempenho final do teste é condicionado a esta base, a esse processo de coleta e a um split fixo. Repetir ajustes vendo o mesmo teste faz o conjunto deixar de funcionar como avaliação externa.

## 10. Plano de execução em etapas

| Etapa | Trabalho | Produto e decisão de passagem |
|---|---|---|
| **G0 — Recuperação/fonte** | Download oficial, SHA-256, descompactação segura, comparar README/termos | `data_manifest.json` completo; conflito de licença registrado |
| **G1 — Auditoria** | Shapes, faltantes, nomes duplicados, classes × participantes, validação de IDs | `dataset_audit.json`, tabelas e testes de invariantes **PASS** |
| **G2 — EDA** | Gráficos F01–F05, recorte do treino | `01_eda.ipynb`; não escolher modelos via teste |
| **G3 — Baselines** | Dummy, logística, k-NN, SVM linear/RBF, RF com `Pipeline` | Scripts e tabela de runtime/baselines agrupados |
| **G4 — Nested CV** | Externo `GroupKFold(5)`, interno `GroupKFold(3)`, previsões OOF | `oof_predictions.csv`, `model_selection.json`, folds auditados |
| **G5 — Hipóteses** | H1 ilustrativo por janelas; H2 seleção/custo; H3 erros | Figuras F06–F13 + conclusões **refutáveis** |
| **G6 — Congelamento** | Fixar modelo, hiperparâmetros, métricas, seeds, hash de código | `protocol_locked.json` assinado por data/commit; **nenhum score de teste ainda** |
| **G7 — Teste final** | Um ajuste no treino completo, uma avaliação dos nove novos sujeitos, bootstrap em grupos | `test_predictions.csv`, `test_metrics.json`, F10/F14; sem retuning |
| **G8 — Artigo/reprodução** | Interpretar limitações, revisar fontes, notebook execução limpa | Artigo técnico em português, README, ambiente e testes reproduzíveis |

**Proposta de organização do texto final:** Resumo; Introdução; Trabalhos relacionados; Dataset e caracterização; Protocolo experimental; Algoritmos/métricas; Resultados (somente depois de computados); Discussão H1–H3; Validade e limites; Conclusão; Referências; Apêndice de reprodução.

## 11. Critérios mensuráveis de aceitação

**Integridade (gates inegociáveis):**

- **100%** dos arquivos essenciais encontrados sob paths reais; ZIP e hashes documentados; CRC sem falha; decomposição legível.
- Exatamente **10.299 linhas** e **561 valores de atributo por janela**, ou divergência formalmente investigada; **seis classes** e nenhuma linha inválida; alinhamento `X/y/subject` **100%**. Valores esperados derivam da documentação e devem ser testados no arquivo [S1, S4–S6].
- Exatamente **21 sujeitos no treino e nove no teste**, **interseção zero**; **zero** presença de `subject_id`/`row_id` em `X`; em **100% dos folds** `set(groups_fit) ∩ set(groups_valid) == ∅`.
- Em **100% dos pipelines** todos os transformadores aprendidos executam `fit` somente no treino correspondente. Um teste unitário comprova que alterar rótulos do teste não altera seleção nem tuning.

**Avaliação e pesquisa:**

- Pelo menos **um Dummy + cinco famílias** (logística, k-NN, SVM linear, SVM RBF, RF) com protocolo homogêneo de grupos; grid e limites documentados.
- `oof_predictions.csv` contém **uma e somente uma previsão para cada uma das 7.352 janelas de treino por modelo avaliado**; `subject_id`, `fold_id`, `y_true`, `y_pred` sem NA.
- Macro-F1, balanced accuracy, confusão 6×6, métricas por classe e sujeito, runtime/latência para cada candidato; **H1, H2 e H3 registradas como confirmadas, contrariadas ou inconclusivas**, sem fabricar evidência.
- Bootstrap por sujeitos com **2.000 réplicas pretendidas**, intervalo/critério de réplicas inválidas registrados; número de clusters divulgado em toda estimativa.
- No mínimo **10 das 14 figuras** do plano, com suporte/denominador, fonte, legenda, acessibilidade e arquivo reprodutível. Exigir obrigatoriamente F02, F03, F04, F07, F09, F10, F11 e F14, salvo impossibilidade justificada.
- Teste oficial usado **apenas uma vez** após bloqueio de protocolo, com logs e hash do modelo. **Nenhum patamar arbitrário de F1 (ex.: 95%) é requisito de aprovação**: método honesto importa mais do que uma pontuação alta.

**Execução e entrega:**

- Alvo 8 GiB RAM/≤2 workers, limite operacional 90 minutos e pico de 6 GiB, **a serem medidos**, não prometidos como verificados.
- Comandos `make audit`, `make run`/equivalentes reproduzíveis em ambiente limpo, dependências fixadas, `pytest` aprovado e saídas idempotentes dentro de tolerância numérica documentada.
- O README deve conter DOI, autores, atribuição CC BY 4.0 **e aviso do conflito com README histórico**, além de nota explícita **“pesquisa de benchmark; não aprovado para uso clínico”**.

## 12. Lista de arquivos esperados no projeto de implementação

```text
har-subject-shift/
├── README.md
├── CITATION.cff
├── LICENSE_DATA_NOTICE.md
├── requirements.in
├── requirements-lock.txt
├── pyproject.toml                 # ou equivalente para pytest/lint
├── data/
│   ├── raw/                     # ZIP oficial, preferencialmente não versionado
│   ├── extracted/               # 'UCI HAR Dataset/' (não versionado)
│   └── processed/               # cache parquet/npz (não versionado)
├── docs/
│   ├── DOSSIER_HAR.md
│   ├── PROTOCOL.md
│   ├── DATA_AUDIT.md
│   ├── MODEL_CARD.md
│   └── LIMITATIONS.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_grouped_cv.ipynb
│   ├── 03_subject_shift_hypotheses.ipynb
│   └── 04_final_report.ipynb
├── src/har_subject_shift/
│   ├── download.py              # fonte oficial, integridade, sem bypass
│   ├── load_data.py             # preserva subject_id e coluna ordinal
│   ├── audit.py
│   ├── splits.py
│   ├── models.py
│   ├── evaluate.py
│   ├── bootstrap.py
│   ├── visualize.py
│   └── cli.py
├── tests/
│   ├── test_data_shapes.py
│   ├── test_no_subject_leakage.py
│   ├── test_pipeline_leakage.py
│   ├── test_oof_integrity.py
│   └── test_bootstrap_by_subject.py
├── artifacts/
│   ├── data_manifest.json
│   ├── dataset_audit.json
│   ├── fold_subjects.json
│   ├── oof_predictions.csv
│   ├── cv_metrics.csv
│   ├── subject_metrics.csv
│   ├── model_selection.json
│   ├── protocol_locked.json
│   ├── test_predictions.csv
│   ├── test_metrics.json
│   ├── figures_manifest.csv
│   ├── runtime.csv
│   └── best_pipeline.joblib
└── figures/
    └── F01_...png a F14_...png (+ SVG quando cabível)
```

**Controle de versão:** não subir os arquivos originais desnecessariamente; documentar como baixá-los. Em artigo público, evitar divulgar `test_predictions.csv` se a divulgação for potencialmente problemática segundo licença/privacidade; ao menos publicar métricas agregadas, código e a regra de reconstrução. **Não usar os artefatos de teste como fonte de tuning.**

## 13. Fontes e referências bibliográficas verificáveis

As fontes são numeradas em ordem de relevância para o protocolo. Links abaixo foram localizados em pesquisa de 21/11/2026; **acesso ao texto** não equivale a download integral dos dados. Os indicadores [O]/[R] declaram o tipo de evidência.

**Datasets — catálogo oficial [O]:**

- **[S1]** Reyes-Ortiz, J.; Anguita, D.; Ghio, A.; Oneto, L.; Parra, X. (2013). *Human Activity Recognition Using Smartphones*. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C54S4K . Página: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones . **Documenta** 10.299 janelas, participantes, sensores, split, ZIP 58,2 MB, licença atual CC BY 4.0.
- **[S2]** Blunck, H.; Bhattacharya, S.; Prentow, T.; Kjærgaard, M.; Dey, A. (2015). *Heterogeneity Activity Recognition*. UCI. DOI: https://doi.org/10.24432/C5689X . Página: https://archive.ics.uci.edu/dataset/344/hhar . **Documenta** usuários, aparelhos, amostras e tamanho anunciado.
- **[S3]** Reiss, A. (2012). *PAMAP2 Physical Activity Monitoring*. UCI. DOI: https://doi.org/10.24432/C5NW2H . Página: https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring . **Documenta** IMUs, participantes, classes, NaN e ZIP.

**Documentação original reproduzida e exemplo de implementação [R]:**

- **[S4]** README original do pacote UCI HAR **reproduzido em repositório de terceiros**: https://github.com/aannasw/uci-har/blob/master/data/UCI-HAR-Dataset/README.txt . **Documenta** nomes/semântica de arquivos, atributos normalizados e aviso histórico de licença. Requer confronto com o pacote obtido diretamente da UCI.
- **[S5]** Anguita, D.; Ghio, A.; Oneto, L.; Parra, X.; Reyes-Ortiz, J. L. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones*. ESANN 2013. Texto dos anais: https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf . **Artigo fundador**.
- **[S6]** `abelraphelp/har-activity-recognition` — *Phase 2: Data Understanding*, relatório de implementação **independente**, não canônico: https://github.com/abelraphelp/har-activity-recognition/blob/main/reports/02_data_understanding.md . **Documenta** IDs por corte e contagens de treinamento, a serem conferidos localmente. Exemplo adicional de projeto clássico, sem usar seus resultados como os nossos: https://github.com/NikolaKujoh/human-activity-recognition .

**Protocolo e literatura:**

- **[S7]** Jordao et al. (2019). *Subject Cross Validation in Human Activity Recognition*. Manuscrito público: https://arxiv.org/abs/1904.02666 . **Demonstra**, em outros dados/condições, estimativas otimistas de validação comum e a importância da validação por sujeito. Não transferir percentuais ao UCI HAR.
- **[S8]** *A Quantitative Comparison of Overlapping and Non-Overlapping Sliding Windows for Human Activity Recognition Using Inertial Sensors* (2019). *Sensors*, 19(22), 5026. https://doi.org/10.3390/s19225026 . Texto: https://www.mdpi.com/1424-8220/19/22/5026 . **Discute** os ganhos aparentes de janelas sobrepostas segundo o protocolo de avaliação.
- **[S9]** *Comparing Person-Specific and Independent Models on Subject-Dependent and Independent Human Activity Recognition Performance* (2020). Texto: https://pmc.ncbi.nlm.nih.gov/articles/PMC7374316/ . **Distingue** teste por pessoa, por tentativa/sessão e repetição de amostras temporais.
- **[S10]** Stisen, A., et al. (2015). *Smart Devices are Different: Assessing and Mitigating Mobile Sensing Heterogeneities for Activity Recognition*. SenSys 2015. https://doi.org/10.1145/2809695.2809718 . Registro acadêmico: https://portal.findresearcher.sdu.dk/en/publications/smart-devices-are-different-assessing-and-mitigating-mobile-sensi/ . **Evidência** de hardware/device shift.
- **[S11]** *Activity-Conditioned Cross-Subject Alignment via Time–Frequency Cross-View Learning for Human Activity Recognition* (2026). *Applied Sciences*, 16(22), 11459. https://www.mdpi.com/2076-3417/16/22/11459 . **Exemplo contemporâneo** de split oficial 21/9 e análise por pessoa; não adotar suas métricas como metas.
- **[S12]** Silva, T. B. C.; Santiago, H. C.; Fernandes, B. J. T. (2026). *From controlled benchmarks to unconstrained environments: a pipeline-oriented survey of models, domains, and emerging trends on human activity recognition*. *Artificial Intelligence Review*. https://doi.org/10.1007/s10462-026-11619-1 . **Revisão** de validade externa e contexto real.
- **[S13]** Biblioteca `stable-datasets`, documentação de UCI HAR: https://galilai-group.github.io/stable-datasets/datasets/uci_har/ . **Registra explicitamente** o conflito entre licença atual UCI e README antigo; não é posicionamento jurídico dos titulares.
- **[S14]** `gholler/UCI-HAR-dataset`, *CodeBook* (réplica/limpeza de terceiros): https://github.com/gholler/UCI-HAR-dataset/blob/master/CodeBook.md . **Alerta** para nomes repetidos em `features.txt` (561 entradas e 477 nomes únicos reportados); confirmar no arquivo original.
- **[S15]** Scikit-learn, *Common pitfalls and recommended practices*: https://scikit-learn.org/stable/common_pitfalls.html e *Pipeline*: https://scikit-learn.org/stable/modules/compose.html . **Base técnica** para prevenir vazamento ao transformar/selecionar atributos.
- **[S16]** *LSTM Networks Using Smartphone Data for Sensor-Based Human Activity Recognition in Smart Homes* (2021). Distribuição de classes e split documentados: https://pmc.ncbi.nlm.nih.gov/articles/PMC7956629/ . **Evidência secundária** de contagens de janelas, não arquivo primário.
- **[S17]** Scikit-learn, *Permutation Importance with Multicollinear or Correlated Features*: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html . **Cautela** em atribuir importância única a features correlacionadas.
- **[S18]** Scikit-learn, `GroupKFold`: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html ; `StratifiedGroupKFold`: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html ; capítulo de validação: https://scikit-learn.org/stable/modules/cross_validation.html . **Protocolo** para grupos não sobrepostos.
- **[S19]** *Internal validation of risk models in clustered data: a comparison of bootstrap schemes* (2013), *American Journal of Epidemiology*. https://pubmed.ncbi.nlm.nih.gov/23660796/ . **Fundamenta** cautela com amostragem bootstrap em estrutura correlacionada.

## 14. O que foi e não foi verificado nesta pesquisa

| Item | Estado em 21/11/2026 |
|---|---|
| Páginas originais UCI, dados catalogados e licenças **indicadas no portal** | **[O] Verificado** |
| Endereços dos botões oficiais de download dos três ZIPs | **[O] Link resolvido**, download integral **[NV]** |
| Bytes do ZIP original UCI HAR; CRC/SHA; conteúdo exato do pacote | **[NV] Não baixado / não auditado** |
| Caminhos `X/y/subject`, nove canais e README | **[R] Documentação histórica reproduzida**, pendente confirmação no ZIP |
| IDs de sujeitos 21/9 e 7.352/2.947 janelas | **[R] Reportados em análise independente e literatura** |
| Contagens totais das seis atividades | **[R] Reproduzidas em literatura**, pendente recálculo direto |
| Contagens por atividade **e por sujeito**; valores faltantes reais; features constantes | **[NV] Não computado** |
| **Tamanho após extração** dos três conjuntos | **[NV] Não mensurado** |
| Divergência entre licença UCI atual e proibição comercial no README | **[O]+[R] Divergência documental constatada**, status legal **não esclarecido** |
| Desempenho de modelos, diferenças H1–H3, bootstrap, latência e gráficos finais | **[P] Não treinados/medidos** |
| Viabilidade real em 8 GiB/90 minutos na máquina de destino | **[P] Meta de engenharia não testada** |

---

## HANDOFF_FOR_IMPLEMENTATION

> **Instrução para a próxima etapa:** produzir código e notebooks executáveis a partir deste protocolo, **sem pular G0/G1**. Nunca confundir números de fontes secundárias com resultados próprios; não usar o teste para escolher modelos.

### Fonte, licença e integridade

- `dataset_id`: **UCI 240 — Human Activity Recognition Using Smartphones**.
- `catalog_url`: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- `doi`: https://doi.org/10.24432/C54S4K
- `official_download_url_seen`: https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip
- `reported_archive_size`: **58,2 MB [O, arredondado no catálogo]**; `actual_sha256`: **PENDENTE**; `extracted_disk_size`: **PENDENTE**; `download_verified_end_to_end`: **NÃO**.
- `license_portal`: **CC BY 4.0 [O]** — citar UCI e autores; `license_archive_readme`: **aviso histórico que proíbe uso comercial [R]**; `commercial_use_clearance`: **NÃO RESOLVIDA**. Nota obrigatória em `LICENSE_DATA_NOTICE.md`.
- `reserve`: **PAMAP2 UCI 231**, sem split pronto, requer janela por sujeito/sessão, NaNs e mais sensores. `extended_benchmark`: **HHAR UCI 344**, múltiplos dispositivos; nunca misturar dados para inferir efeito apenas de sujeito sem controlar dispositivo.

### Caminhos, schema e grupos

```yaml
root_expected: "data/extracted/UCI HAR Dataset"  # verificar via zipfile, NÃO presumir sem teste
train_X: "train/X_train.txt"
train_y: "train/y_train.txt"
train_subject: "train/subject_train.txt"
test_X: "test/X_test.txt"
test_y: "test/y_test.txt"
test_subject: "test/subject_test.txt"
feature_names: "features.txt"
feature_info: "features_info.txt"
activity_names: "activity_labels.txt"
raw_inertial_optional: "{train,test}/Inertial Signals/"
expected_shape_train: [7352, 561]  # [R] checar
expected_shape_test: [2947, 561]   # [R] checar
expected_subjects_train: [1,3,5,6,7,8,11,14,15,16,17,19,21,22,23,25,26,27,28,29,30]  # [R]
expected_subjects_test: [2,4,9,10,12,13,18,20,24]                       # [R]
subject_overlap_allowed: 0
subject_id_as_predictor: false
labels: {1: WALKING, 2: WALKING_UPSTAIRS, 3: WALKING_DOWNSTAIRS, 4: SITTING, 5: STANDING, 6: LAYING}
canonical_feature_ids: "f001..f561, in original order; original names metadata"
```

### Protocolo irrevogável antes do teste

```yaml
seed: 42
primary_metric: "macro-F1, six labels fixed"
secondary_metrics: [balanced_accuracy, per_class_precision_recall_f1, per_subject_macro_f1, confusion_matrix]
outer_validation: "GroupKFold(n_splits=5), groups=subject_train"
inner_tuning: "GroupKFold(n_splits=3), only outer-training subjects"
optional_stratification: "StratifiedGroupKFold only if subject × class audit motivates it"
illustrative_incorrect_protocol: "StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on official train ONLY; fixed untuned configuration"
final_external_test: "Official UCI test, 9 unseen subjects; one-time evaluation after protocol lock"
models: [DummyMostFrequent, MultinomialLogistic, StandardizedKNN, StandardizedLinearSVC, StandardizedRBFSVC, RandomForest]
feature_selection_ablation: "VarianceThreshold inside pipelines; SelectKBest k=64,128,256,all only on train folds"
bootstrap_resamples: 2000
bootstrap_unit: "subject_id with replacement, whole-window blocks, paired for model differences"
ci: "percentile 2.5%..97.5%, warning n_subjects_test=9"
training_cpu_workers_max: 2
ram_target_gib: 8
process_peak_target_gib: 6
wall_time_target_minutes: 90
nn_required: false
required_invariants: [no_subject_intersection, one_oof_prediction_per_train_row, no_preprocessing_test_fit, test_not_used_for_selection]
```

### Saídas obrigatórias e critérios de interrupção

- **Antes de qualquer modelo:** `data_manifest.json`, `dataset_audit.json`, `fold_subjects.json`; **interromper** se arquivos/linhas/alinhamento, separação de sujeitos ou licença não estiverem documentados.
- **Antes de abrir `test/`:** `oof_predictions.csv`, `cv_metrics.csv`, `subject_metrics.csv`, `model_selection.json` e `protocol_locked.json`; **interromper** se algum `subject_id` aparecer em treino e validação do mesmo fold, ou se seletores usarem o teste.
- **Depois de teste único:** `test_predictions.csv`, `test_metrics.json`, intervalos por sujeitos, F01–F14 previstas, `runtime.csv`, `best_pipeline.joblib` e artigo com limites e referências.
- **Critério científico de sucesso:** avaliar H1/H2/H3 com documentação verificável e possibilidade de **refutação**, independentemente da pontuação dos modelos. **Nunca escrever métricas não observadas.**

### Pendências explícitas para o executor

1. Baixar ZIP **da UCI**, verificar CRC, SHA-256, tamanho extraído e a estrutura verdadeira; confrontar `README.txt` e termos de uso, preservando a divergência de licença.
2. Confirmar no arquivo **todos** os IDs e distribuições de classes/sujeitos acima; confirmar que não há valores não finitos, nomes/colunas trocados e interseção de sujeitos.
3. Definir a partir da auditoria se algum sujeito não tem todas as classes e como tratar a métrica individual com `labels` fixos; guardar decisão antes do teste.
4. Medir tempos reais, convergência, picos de memória e qualidade dos folds; adaptar só os **grids** dentro do treino se o orçamento falhar, registrando mudança no protocolo.
5. Confirmar conformidade de redistribuição/publicação do dataset; para **uso comercial**, pedir esclarecimento jurídico/titulares por causa da contradição documental.
6. Obter os resultados reais e redigir conclusão sem “pré-confirmar” H1, H2 ou H3.

**Fim do dossiê.**
