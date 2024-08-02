# Deep Learning-based Model for Hepatotoxicity Prediction

### 0. Environmental settings
```
    pip install -r requirements.txt
```

### 1. Get hepatotoxicity score (and uncertainty) from model
```
    python main.py --smiles_filename sample/sample.smi
                   --result_filename sample/sample.out
```
or, you can use a single SMILES
```
    python main.py --smiles "CN(C)[C@H]1[C@@H]2C[C@@H]3CC4=C(C=CC(=C4C(=C3C(=O)[C@@]2(C(=C(C1=O)C(=O)N)O)O)O)O)N(C)C"
                   --result_filename sample/minocycline.out
```

### 2. Visualizing atom contribution (support for a single SMILES)
```
    python main.py --smiles "CN(C)[C@H]1[C@@H]2C[C@@H]3CC4=C(C=CC(=C4C(=C3C(=O)[C@@]2(C(=C(C1=O)C(=O)N)O)O)O)O)N(C)C"
                   --result_filename sample/minocycline.out
                   --image_filename draw/minocycline.png
```

<img src="https://github.com/WonhoZhung/HEPATOTOXICITY_PREDICTION/blob/main/draw/minocycline.png" height="400">


### 3. Test set benchmark score
```
+--------------------+----------+-------+----------+
| Test set benchmark | Accuracy | AUROC | F1 Score |
+--------------------+----------+-------+----------+
| Previous model     | 0.591    | 0.646 | 0.636    |
+--------------------+----------+-------+----------+
| Current model      | 0.717    | 0.745 | 0.798    |
+--------------------+----------+-------+----------+
```
