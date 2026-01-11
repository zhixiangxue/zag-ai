# Pre-Release Testing

## Quick Start

Run this before publishing to PyPI:

```powershell
.\scripts\pre_release.ps1
```

## What It Does

1. Creates a clean temporary virtual environment
2. Installs `zagpy` from your project directory
3. Runs a complete E2E RAG pipeline test
4. Cleans up automatically

## Prerequisites

### Required: Ollama

The test uses Ollama for local embeddings (no cloud API needed).

**Install Ollama** (one-time setup):

```bash
# Windows
# Download from: https://ollama.ai/download/windows

# Linux/WSL
curl -fsSL https://ollama.com/install.sh | sh

# Mac
brew install ollama
```

**Pull the embedding model** (one-time setup):

```bash
ollama pull nomic-embed-text
```

**Verify it's working**:

```bash
ollama list
# Should show: nomic-embed-text
```

## What Gets Tested

- ✅ PDF document reading (DoclingReader)
- ✅ Text splitting & chunking (RecursiveMergingSplitter)
- ✅ Embedding generation (Ollama local)
- ✅ Vector indexing (ChromaDB local)
- ✅ Semantic retrieval (VectorRetriever)
- ✅ Result postprocessing (SimilarityFilter)

## No External Services Needed

- ❌ No Meilisearch
- ❌ No Qdrant server
- ❌ No cloud APIs or API keys
- ✅ Everything runs locally

## Troubleshooting

### Error: "Failed to create embedder"

**Solution**: Install and start Ollama

```bash
# 1. Install Ollama (see above)
# 2. Pull model
ollama pull nomic-embed-text
# 3. Verify
ollama list
```

### Error: "Package installation failed"

**Solution**: Check your `pyproject.toml` for missing dependencies

### Test passes but you see warnings

Warnings from dependencies (like docling) are normal and don't affect functionality.

## CI/CD Integration

You can run the test script in CI/CD, but you'll need to:

1. Install Ollama in the CI environment
2. Pull the nomic-embed-text model
3. Run the script

Example GitHub Actions:

```yaml
- name: Setup Ollama
  run: |
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull nomic-embed-text

- name: Run pre-release test
  run: |
    pwsh ./scripts/pre_release.ps1
```

## After Tests Pass

Your package is ready to publish:

```bash
# Build
python -m build

# Upload to PyPI
twine upload dist/*
```
