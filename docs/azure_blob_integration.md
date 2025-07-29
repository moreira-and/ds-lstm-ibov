# Alterações para Integração com Azure Blob Storage

## Visão Geral
Este documento descreve as alterações realizadas para integrar o projeto com Azure Blob Storage,
seguindo uma arquitetura de dados em camadas (medallion architecture).

## Estrutura de Diretórios
Criada nova estrutura de diretórios para melhor organização:
```
src/
└── dataset/
    ├── loaders/         # Carregadores de dados de diferentes fontes
    ├── uploaders/       # Gerenciadores de upload para diferentes storages
    └── helpers/         # Funções auxiliares
```

## Novos Componentes

### 1. AzureBlobStorageLoader (`src/dataset/uploaders/azure_blob.py`)
- **Propósito**: Gerenciar operações de leitura/escrita no Azure Blob Storage
- **Funcionalidades**:
  - `save_to_layer`: Salva dados em uma camada específica (raw, bronze, silver)
  - `load_from_layer`: Carrega dados de uma camada específica
  - Suporte para formatos parquet e CSV
  - Implementa a interface `IDatasetLoaderStrategy`

### 2. Novos Loaders de Dados
- **CVMLoader**: Extrai dados de fundos da CVM
  - URL: 'https://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv'
  - Salva em formato CSV na camada raw

- **IBGELoader**: Extrai dados do IPCA
  - URL: 'https://apisidra.ibge.gov.br/values/t/1737/n1/all/v/2266/p/all/d/v2266%202'
  - Salva em formato JSON na camada raw

## Modificações em Arquivos Existentes

### 1. `run_dataset.py`
- Removido salvamento redundante em arquivos locais
- Integrado com Azure Blob Storage
- Adicionados novos loaders (CVM e IBGE)
- Fluxo modificado:
  1. Extrai dados das fontes
  2. Salva na camada raw do blob
  3. Combina dados na camada silver
  4. Enriquece com dados de calendário
  5. Salva versão final na camada silver

### 2. `run_features.py`
- Modificado para ler dataset do blob ao invés de arquivo local
- Mantido processamento e salvamento local de features e transformers
- Fluxo:
  1. Carrega dataset enriquecido da camada silver do blob
  2. Processa features normalmente
  3. Salva arquivos processados localmente

### 3. `configs/dataset.yaml`
- Adicionado código da poupança (195) à configuração do BCB:
```yaml
bcb:
  SELIC: 11
  CDI: 12
  SELIC_Anual: 1178
  poupanca: 195
```

## Configuração
Adicionadas variáveis de ambiente necessárias (.env):
```
AZURE_STORAGE_CONNECTION_STRING=your_connection_string_here
AZURE_STORAGE_CONTAINER_NAME=your_container_name_here
```

## Arquitetura de Dados
Implementada arquitetura em camadas no Azure Blob Storage:
1. **Raw**: Dados brutos extraídos das fontes
   - Arquivos individuais de cada fonte
   - Formatos originais (CSV, JSON)

2. **Silver**: Dados processados e combinados
   - Dataset combinado e enriquecido
   - Features processadas
   - Transformers serializados

## Dependências Adicionadas
- Azure Storage Blob (`azure-storage-blob==12.19.0`)
- Atualizado `requirements.txt`

## Como Usar
1. Configure as variáveis de ambiente do Azure Blob Storage
2. Execute `run_dataset.py` para extrair e combinar dados
3. Execute `run_features.py` para processar features

## Observações
- Mantida compatibilidade com salvamento local como fallback
- Implementada gestão de erros e logging
- Código modular permite fácil adição de novos uploaders no futuro
