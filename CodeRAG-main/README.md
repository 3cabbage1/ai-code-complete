# CodeRAG

## 📦 Environment Setup

### 1. **Install [uv](https://docs.astral.sh/uv/)**
### 2. **Synchronize dependencies**
   ```bash
   uv sync
   ```
### 3. **Activate the virtual environment**
   ```bash
   source .venv/bin/activate
   ```

---

## 🚀 Usage

Before running scripts, download benchmarks (recceval and cceval) and edit the configuration file:

```bash
config/config.toml
```

Then execute the Python scripts **sequentially**:

### 1. Build Query
```bash
python scripts/build_query.py
```
- Generates query strings from the benchmark dataset.

### 2. Retrieve Relevant Code Blocks
```bash
python scripts/retrieve.py
```
- Retrieves top-k relevant code blocks using the configured retriever.



### 3. Build Prompts for Generator
```bash
python scripts/build_prompt.py
```
- Constructs prompts from retrieved code blocks for the code completion generator.

### 4. Run Inference
```bash
python scripts/inference.py
```
- Feeds prompts to the generator model.
- You can replace this step with your own inference code.  
  **Input:** JSON file containing an array of strings  
  **Output:** JSON file containing an array of generated completions.


