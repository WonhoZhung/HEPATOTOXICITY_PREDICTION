# Deep-learning Model for Hepatotoxicity Prediction

### 0. Environmental settings
```
    pip install -r requirements.txt
```

### 1. Get hepatotoxic score from model
```
    python main.py --model_saved model/best_model.pt
                   --smiles_filename sample/sample.smi
                   --result_filename sample.out
```

### 2. Visualizing atom contribution
```
    python main.py --model_saved model/best_model.pt
                   --smiles_filename sample/minocycline.smi
                   --result_filename minocycline.out
                   --visualize
```

<img src="https://github.com/WonhoZhung/HEPATOTOXICITY_PREDICTION/blob/main/draw/minocycline.png" height="400">
