# Dez perguntas de entrevista — HAR sujeito-independente

**1. Por que grupos por pessoa?** Janelas de um mesmo voluntário compartilham características de movimento e amostras temporais, logo não estimam corretamente desempenho em pessoas inéditas quando espalhadas entre treino e validação.

**2. Por que macro-F1?** Faz média do F1 nas seis classes com pesos iguais, impedindo que classes frequentes dominem o resumo. Fixamos sempre labels 1–6.

**3. O que significa subject shift?** A distribuição de sinais muda quando o portador do smartphone muda: marcha, amplitude e velocidade variam entre indivíduos.

**4. O split aleatório produz leakage sempre?** Ele mistura pessoas; por isso mede outra tarefa. Também pode misturar janelas vizinhas sobrepostas. O contraste não separa causalmente as fontes de otimismo.

**5. Por que pipeline?** Scaler, remoção de variância e seleção aprendem parâmetros somente nos folds de ajuste, não nos folds de validação/holdout.

**6. O que é validação aninhada?** Fold externo avalia configurações escolhidas em CV interna com pessoas disjuntas; reduz a contaminação da estimativa de seleção de hiperparâmetros.

**7. Como quantificar incerteza?** Bootstrap com reposição de IDs de pessoas inteiras. Com apenas nove pessoas no teste, o IC é descritivo e instável.

**8. Quais modelos são baselines?** Dummy most-frequent, regressão logística, kNN, SVM linear e RBF, Random Forest. A comparação usa protocolos homogêneos e tempo de CPU registrado.

**9. Como analisar erros por atividade?** Matriz de confusão 6×6 normalizada por atividade real, recall por classe e taxas de erros estáticos vs dinâmicos com denominadores explícitos.

**10. Por que não retunar após o teste?** Porque o teste deixaria de ser independente. O arquivo `test_started.json` bloqueia uma segunda execução silenciosa, enquanto `protocol_locked.json` preserva as decisões anteriores.
