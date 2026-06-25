# Guia — LangFuse local no Windows via Podman

Este guia sobe o **LangFuse** (observabilidade dos pipelines RAG) na sua máquina,
usando o **Podman** (o mesmo que você já instalou na Aula 1) com o `docker-compose.yml`
oficial do projeto. O objetivo é só **baixar e executar** — não vamos editar nem
explicar os parâmetros do compose.

Referência oficial: https://langfuse.com/self-hosting/deployment/docker-compose

---

## 1. Pré-requisitos

- **Podman** instalado e a máquina Podman iniciada (Aula 1). Para conferir:
  ```powershell
  podman machine list
  ```
  Se não estiver rodando, inicie com `podman machine start`.
- **git** instalado (para baixar o projeto). Alternativa sem git: baixar o ZIP do
  repositório em https://github.com/langfuse/langfuse (botão **Code → Download ZIP**).

---

## 2. Baixar o LangFuse

No PowerShell, em uma pasta de sua preferência:

```powershell
git clone https://github.com/langfuse/langfuse.git
cd langfuse
```

> Sem git? Baixe o ZIP, extraia, e entre na pasta `langfuse` extraída.

---

## 3. Executar via Podman

Dentro da pasta `langfuse` (onde está o arquivo `docker-compose.yml`):

```powershell
podman compose up -d
```

- O `-d` roda em segundo plano.
- Na **primeira vez** o Podman baixa as imagens (alguns GB) — pode levar alguns minutos.
- Aguarde ~2–3 minutos até o serviço web ficar pronto.

Para acompanhar os logs (opcional):

```powershell
podman compose logs -f
```

> Se o comando `podman compose` não existir na sua versão, instale o `podman-compose`
> (dentro do venv: `pip install podman-compose`) e use `podman-compose up -d`.

---

## 4. Acessar e criar as chaves

1. Abra no navegador: **http://localhost:3000**
2. Crie uma conta local (e-mail e senha quaisquer — é só na sua máquina).
3. Crie uma **Organization** e um **Project**.
4. No projeto, vá em **Settings → API Keys → Create new API keys**.
5. Copie a **Public Key** e a **Secret Key**.

---

## 5. Configurar o `.env` do curso

No `.env` (raiz do projeto `MBA_RAG_CAG`), preencha:

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=http://localhost:3000
```

Pronto. Quando essas chaves existem, o script `04_advanced_rag.py` liga a
auto-instrumentação sozinho e mostra, ao final, o link do trace no LangFuse.

> Verifique tudo com: `python 00_check_ambiente.py`

---

## 6. Parar / iniciar de novo

```powershell
podman compose stop      # para os conteineres (mantem os dados)
podman compose up -d     # inicia novamente
podman compose down      # remove os conteineres (mantem os volumes/dados)
```

---

*MBA RAG & CAG Aplicados a Direito e Segurança Pública — Aula 3*
