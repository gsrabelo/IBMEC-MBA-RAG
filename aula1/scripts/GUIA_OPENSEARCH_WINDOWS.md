# Guia — OpenSearch single-node por binário no Windows (com kNN)

Este guia sobe o **OpenSearch** direto pelo binário (`.zip`), **sem Docker e sem Podman**,
em modo de **um nó só (single-node)** e com a **busca vetorial kNN** pronta para os scripts
da Aula 1. O plugin `opensearch-knn` **já vem incluído** na distribuição — não precisa instalar à parte.

> Versão usada neste guia: **OpenSearch 3.6.0** (a mais recente em junho/2026).
> Os passos valem para qualquer versão 3.x.

---

## 1. Baixar

1. Acesse a página oficial de downloads: https://opensearch.org/downloads/
2. Na seção **OpenSearch**, baixe o arquivo do Windows:
   `opensearch-3.6.0-windows-x64.zip`

> O `.zip` é autocontido: já traz o Java (JDK) embutido. Você **não** precisa instalar Java.

---

## 2. Extrair

1. Clique com o botão direito no `.zip` → **Extrair tudo**.
2. Extraia para um caminho **sem espaços e sem acentos**. Recomendado:
   ```
   C:\opensearch
   ```
   (o resultado deve ficar tipo `C:\opensearch\opensearch-3.6.0\`)

> ⚠️ Caminho com espaço/acento faz o OpenSearch **não iniciar**. Evite a Área de Trabalho.

---

## 3. Configurar para o laboratório (single-node, sem HTTPS/senha)

Para o curso, vamos rodar de forma simples: um nó só e **sem o plugin de segurança**
(sem HTTPS e sem usuário/senha). Isso facilita os scripts em `localhost:9200`.

1. Abra o arquivo de configuração em um editor de texto (ex.: VS Code, Bloco de Notas):
   ```
   C:\opensearch\opensearch-3.6.0\config\opensearch.yml
   ```
2. Adicione estas linhas no final do arquivo e salve:
   ```yaml
   # Modo de nó único (lab)
   discovery.type: single-node

   # Desliga o plugin de seguranca (sem HTTPS e sem senha) - APENAS para o lab
   plugins.security.disabled: true

   # Aceita conexoes locais
   network.host: 0.0.0.0
   ```

> 🔒 **Importante:** `plugins.security.disabled: true` é só para ambiente de estudo na sua
> máquina. Em produção (TJ, MP, PF) a segurança fica **ligada** com HTTPS e credenciais.

---

## 4. Iniciar o OpenSearch

1. Abra o **PowerShell** (ou Prompt de Comando).
2. Entre na pasta extraída:
   ```powershell
   cd C:\opensearch\opensearch-3.6.0
   ```
3. Inicie o servidor:
   ```powershell
   .\opensearch-windows-install.bat
   ```
   - Use esse `.bat` na **primeira vez**. Nas próximas, pode usar `.\bin\opensearch.bat`.
   - Se o Windows perguntar sobre o **Firewall**, libere o acesso.
4. **Deixe essa janela aberta** — o OpenSearch fica rodando nela. Aguarde ~30–60 segundos
   até aparecer algo como `... started`.

---

## 5. Verificar se está no ar

Abra **outro** PowerShell e rode:

```powershell
curl http://localhost:9200
```

Você deve ver um JSON com `"cluster_name"` e a versão. Para conferir o **kNN**:

```powershell
curl http://localhost:9200/_cat/plugins?v
```

Procure por uma linha contendo `opensearch-knn` na lista. Se aparecer, a busca vetorial está pronta.

> Atalho: o script `python 00_check_ambiente.py` faz essas duas verificações para você.

---

## 6. Pronto — próximos passos

Com o OpenSearch no ar e o Ollama rodando, volte ao `README_SCRIPTS.md` e execute:

```bash
python 04_indexar_opensearch.py        # cria o indice kNN e indexa o corpus
python 05_rag_minimo.py --pergunta "Quais os requisitos da prisao preventiva?"
```

---

## 7. Problemas comuns

| Sintoma | Causa provável | Solução |
|---|---|---|
| Não inicia / fecha sozinho | Caminho com espaço/acento | Reextraia em `C:\opensearch` |
| `curl` não responde | Ainda subindo, ou janela fechada | Aguarde e mantenha a janela do passo 4 aberta |
| Pede usuário/senha | Segurança ainda ligada | Confira `plugins.security.disabled: true` no `opensearch.yml` e reinicie |
| `opensearch-knn` não aparece | Distribuição incompleta | Rebaixe o `.zip` oficial e reextraia |
| Erro de memória (heap) | Pouca RAM disponível | Em `config\jvm.options` ajuste `-Xms1g` / `-Xmx1g` |

---

*MBA RAG & CAG Aplicados a Direito e Segurança Pública — Aula 1*
