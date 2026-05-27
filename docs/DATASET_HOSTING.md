# Dataset Hosting — HuggingFace Hub

The raw `creditcard.csv` (144 MB) and converted `creditcard.parquet` (46 MB)
are **never committed to this repository**.  The Streamlit app downloads the
dataset at runtime from a HuggingFace Hub dataset repo.

---

## Why HuggingFace Hub?

| Provider | Cost | Auth | Git-native | ML-native | Direct URL |
|---|---|---|---|---|---|
| **HuggingFace Hub** | Free | Optional | ✅ | ✅ | ✅ |
| Google Drive | Free | OAuth2 headache | ❌ | ❌ | ✗ |
| AWS S3 | ~$0.023/GB | IAM keys | ❌ | ❌ | ✅ |
| Supabase Storage | Free tier | JWT | ❌ | ❌ | ✅ |

HuggingFace is the standard for ML portfolio projects — datasets are
versioned, citable, and directly readable with `pd.read_parquet(url)`.

---

## One-time upload (do this once, then the app downloads automatically)

### Prerequisites
```bash
pip install huggingface_hub
huggingface-cli login        # enter your HF token when prompted
```

### 1. Convert CSV → Parquet (if not already done)
```bash
python - <<'EOF'
import pandas as pd

df = pd.read_csv("data/creditcard.csv")
for col in [c for c in df.columns if c.startswith("V")]:
    df[col] = df[col].astype("float32")
df["Amount"] = df["Amount"].astype("float32")
df["Time"]   = df["Time"].astype("int32")
df["Class"]  = df["Class"].astype("int8")
df.to_parquet("data/creditcard.parquet", index=False, compression="zstd")
print(f"Saved {len(df):,} rows, "
      f"{__import__('os').path.getsize('data/creditcard.parquet')/1e6:.1f} MB")
EOF
```

### 2. Create the HuggingFace dataset repo
Go to https://huggingface.co/new-dataset and create a repo named
`credit-card-fraud` under your account.  Set it to **Public** (free plan).

### 3. Upload the Parquet file
```bash
huggingface-cli upload ParthPardeshi18/credit-card-fraud \
    data/creditcard.parquet creditcard.parquet \
    --repo-type dataset
```

### 4. Verify the download URL
Open in your browser:
```
https://huggingface.co/datasets/ParthPardeshi18/credit-card-fraud/resolve/main/creditcard.parquet
```
You should see a download prompt for a ~46 MB file.

### 5. Update `src/data_pipeline.py`
```python
HF_DATASET_REPO = "ParthPardeshi18/credit-card-fraud"   # ← your username/repo
HF_PARQUET_FILE = "creditcard.parquet"
```

---

## How the app loads data at runtime

```
fraud_app.py  ──  st_load_dataframe()   ← @st.cache_data, 24 h TTL
                        │
                  load_dataframe()
                        │
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
    local parquet   disk cache    HuggingFace
    (dev only)     (~/.../tmp)    HTTPS download
```

The first cold start on Streamlit Cloud downloads ~46 MB once, caches it to
`tempfile.gettempdir()`, and serves every subsequent rerun from memory
(courtesy of `@st.cache_data`).

---

## Streamlit Cloud secrets (optional — only for private HF repos)

In the Streamlit Cloud dashboard → **App settings → Secrets**, add:
```toml
HF_TOKEN = "hf_your_read_only_token_here"
```

The token is read by `data_pipeline._read_streamlit_secret()` and passed as
an `Authorization` header during download.

---

## Local development (no internet required)

Place either file in `data/` and the pipeline auto-detects it:
```
data/creditcard.parquet   ← preferred (fastest, 3× smaller)
data/creditcard.csv       ← also supported (legacy)
```
Both paths are in `.gitignore` — they will never be accidentally committed.
