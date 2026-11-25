# Limitações e ameaças à validade

1. Pessoas e janelas não são independentes; bootstrap por janelas é inadequado.
2. Cinco folds compartilham dados de treinamento; intervalos bootstrap de OOF são descritivos, não inferência causal.
3. Nove indivíduos no teste oficial são poucos clusters, e resultados variam com a composição dos sujeitos.
4. Os dados vêm de um celular na cintura em atividades roteirizadas; não são evidência para relógios, rotina livre ou diagnóstico.
5. Separação por pessoa não isola exclusivamente o efeito de janelas sobrepostas; diferenças de estilo corporal e sessões podem contribuir.
6. Seleção de família vencedora a partir de métricas externas de validação tem winner's curse. O holdout único é a estimativa final separada.
7. Métricas agregadas por janelas pesam mais indivíduos com mais janelas; métricas por pessoa com seis classes fixadas mitigam essa interpretação.
8. Conflito de licença do portal UCI vs README original está aberto.
9. Métricas reais exigem baixar, inspecionar e executar dataset oficial; indisponível no ambiente do pacote.
10. Modo quick não faz tuning comparável ao grid full, devendo ser claramente rotulado como simplificação.
