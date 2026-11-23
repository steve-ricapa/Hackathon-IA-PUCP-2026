# data/

Aquí va la data de la competencia. **No se sube a GitHub** (está en el `.gitignore`).

Bájala de la pestaña *Data* de la competencia en Kaggle o con la API, desde la raíz del repositorio:

```bash
kaggle competitions download -c hackathon-ia-pucp-2026
unzip hackathon-ia-pucp-2026.zip -d data/
```

Debe quedar así:

```text
data/
├── train.npz
├── test.npz
├── static.npz
└── sample_submission.csv
```
