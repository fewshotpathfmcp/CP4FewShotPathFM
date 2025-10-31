# Towards Reliable Few-Shot Adaptation of Pathology Foundation Models via Conformal Prediction


## Dataset Guide

### Datasets Used
- **HiCervix Dataset** : [Download Link](https://zenodo.org/records/11087263)
- **HMU‑GC‑HE‑30K Dataset**: [Download Link](https://figshare.com/articles/dataset/Gastric_Cancer_Histopathology_Tissue_Image_Dataset_GCHTID_/25954813)

### Embeddings: [Download Link](https://drive.google.com/drive/folders/1aYn_sCpJyRFr3JUXvn9n0jXBCIYcDj-o?usp=sharing)

## Experiments 
Run any experiment script directly from your terminal using the below:

### Baseline
```bash
python baseline.py --embeddings "C:\path\to\embeddings.pt"
```

### Baseline++ 
```bash
python baseline++.py --embeddings "C:\path\to\embeddings.pt"
```

### Prototype Networks
```bash
python prototype.py --embeddings "C:\path\to\embeddings.pt"
```


Notes:
- The `--embeddings` (or `-e`) argument is optional. If omitted, each script searches common default paths.
- Outputs include a timestamped JSON and CSV in the current directory.
