## CP4FewShotPathFM 

Run any experiment script directly from your terminal. From this folder:

### Prototype Networks
```bash
python prototype.py --embeddings "C:\path\to\embeddings.pt"
```

### Baseline++ Cosine Classifier
```bash
python baseline++.py --embeddings "C:\path\to\embeddings.pt"
```

### Logistic Regression Baseline
```bash
python baseline.py --embeddings "C:\path\to\embeddings.pt"
```

Notes:
- The `--embeddings` (or `-e`) argument is optional. If omitted, each script searches common default paths.
- Outputs include a timestamped JSON and CSV in the current directory.
