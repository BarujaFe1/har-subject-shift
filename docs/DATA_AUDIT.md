# Auditoria do dataset

- `har-shift acquire`: só baixa da UCI 240, em ZIP temporário e promove após CRC. Diretório `data/raw` não versionado.
- `har-shift audit`: inspeciona entradas e CRC do ZIP, rejeita caminhos `..`, absolutos e links simbólicos. Exige arquivos básicos, extrai com segurança, calcula SHA-256 e bytes reais.
- Verifica shapes esperados `(7352, 561)`/`(2947,561)`, classes 1–6, IDs 1–30, 21/9 participantes disjuntos, ausência de NaN/Inf, ordenação dos 561 nomes sem eliminar duplicados, rótulos e alinhamento das colunas.
- Calcula duplicatas exatas de vetores dentro dos cortes e entre cortes, contagem de classes, tabela classe × sujeito e verifica cláusula de licença na cópia baixada.
- Conferir `artifacts/dataset_audit.json` e `data_manifest.json` após execução real. Não há tais resultados neste pacote porque download não foi possível.

A auditoria lê o conjunto de teste **apenas para integridade estrutural**. Não ajusta modelo nem usa desempenho do teste para decisões.
