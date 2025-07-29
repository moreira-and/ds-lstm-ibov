# Changelog - Integração Azure Blob Storage

## [0.1.0] - 2025-07-27

### Adicionado
- Novo módulo `uploaders` para gerenciamento de storage
- Classe `AzureBlobStorageLoader` para operações no Azure Blob Storage
- Suporte para arquitetura de dados em camadas (raw, silver)
- Novos loaders para CVM e IBGE
- Documentação detalhada da integração

### Modificado
- `run_dataset.py`: integrado com blob storage
- `run_features.py`: leitura de dados do blob
- Configuração BCB: adicionado código da poupança
- Requirements: adicionada dependência azure-storage-blob

### Removido
- Salvamento redundante de arquivos na pasta raw

### Técnico
- Reorganização da estrutura de diretórios
- Implementação de interface IDatasetLoaderStrategy
- Adição de variáveis de ambiente para configuração
