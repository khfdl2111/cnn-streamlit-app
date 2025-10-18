# CNN + Streamlit Starter

## 1) Persiapkan data
```
data/
  train/class_a ...
  val/class_a   ...
  test/class_a  ... (opsional untuk evaluasi)
```

## 2) Training
```
# tanpa tuning
python train_transfer.py --data_dir data --epochs 10

# dengan tuning
python train_transfer.py --data_dir data --tune --max_epochs 15
```
Output: `artifacts/model.keras`, `artifacts/class_names.json`

## 3) Jalankan aplikasi Streamlit
```
streamlit run app.py
```

## 4) Evaluasi pada data baru
```
python evaluate.py --test_dir data/test --artifacts_dir artifacts
```
Tersimpan: `artifacts/eval_report.json`, `artifacts/confusion_matrix.png`
