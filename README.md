## CP4FewShotPathFM 

Run any experiment script directly from your terminal. From this folder:

### Prototype Networks
```bash
python prototype.py --embeddings "C:\path\to\embeddings.pt"
```

### Cosine Classifier
```bash
python COSINE.PY --embeddings "C:\path\to\embeddings.pt"
```

### Logistic Regression Baseline
```bash
python Baseline.py --embeddings "C:\path\to\embeddings.pt"
```

Notes:
- The `--embeddings` (or `-e`) argument is optional. If omitted, each script searches common default paths.
- Outputs include a timestamped JSON and CSV in the current directory.