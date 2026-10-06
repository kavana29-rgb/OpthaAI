# Diabetic Retinopathy Research Assistant

This repository is a **research and education prototype**, not a medical device. It demonstrates an end-to-end image-classification workflow for five diabetic-retinopathy grades from color fundus photographs:

| Label | Meaning |
| --- | --- |
| `0` | No diabetic retinopathy |
| `1` | Mild |
| `2` | Moderate |
| `3` | Severe |
| `4` | Proliferative diabetic retinopathy |

The model is a pretrained ImageNet ResNet-50 fine-tuned with class-weighted cross-entropy. The application includes black-border cropping, augmentation, balanced accuracy and macro-F1 evaluation, and Grad-CAM explanations. It intentionally refuses to produce a prediction until a real trained checkpoint exists.

## 1. Data and clinical safeguards

Use a dataset whose license permits your intended use, such as Kaggle's **APTOS 2019 Blindness Detection** or EyePACS-derived data. Do not upload identifiable patient information. Keep patient/eye-level groups together when splitting so images from one patient cannot leak across train, validation, and test sets.

Create this folder structure after downloading and converting the dataset:

```text
data/processed/
  train/{0,1,2,3,4}/*.jpg
  val/{0,1,2,3,4}/*.jpg
  test/{0,1,2,3,4}/*.jpg
```

Never evaluate only on a random image split when patient IDs are available. Report per-class sensitivity, specificity, confusion matrix, macro-F1, balanced accuracy, calibration, and confidence intervals. Validate on an external camera/site population before any clinical use. Human expert review remains mandatory.

## 2. Kaggle or Colab training

In a Kaggle notebook, enable GPU, clone/upload this repository, install dependencies, prepare `data/processed`, and run:

```bash
pip install -r requirements.txt
python train.py --data-dir data/processed --epochs 10 --batch-size 16
```

In Colab, upload the repository or clone it, mount Drive if the dataset/checkpoint is stored there, and run the same commands. The best validation macro-F1 checkpoint is written to `models/dr_resnet50.pth`. Download that file and place it in the local `models/` directory.

Evaluate the untouched test split only after model selection:

```bash
python evaluate.py --data-dir data/processed --checkpoint models/dr_resnet50.pth
```

This writes `artifacts/test_metrics.json` with balanced accuracy, macro-F1, per-class precision/recall/F1, and a confusion matrix. The current scripts are a reproducible baseline, not a validated clinical training recipe. For a serious study, add patient-grouped splitting, external validation, confidence intervals, subgroup analysis, threshold selection on validation data only, probability calibration, and prospective reader comparison.

## 3. Run locally

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit. The app shows the input, class probabilities, and a Grad-CAM overlay. It does not save uploads or send them to a remote API.

## 4. Deploy on Streamlit Community Cloud

1. Push this repository and the trained checkpoint to a private or appropriately governed GitHub repository. Do not commit patient images or secrets.
2. In Streamlit Community Cloud, select `app.py` as the main file.
3. Use Python 3.11 and the `requirements.txt` file.
4. Large model files may exceed GitHub limits; use an approved private artifact store and download the checkpoint during deployment with access controls. Never place credentials in source code.
5. Keep the warning, model card, dataset license, version, and validation metrics visible to users.

## Limitations and model card summary

This prototype can fail on poor focus, blur, non-retinal images, different cameras, uncommon pigmentation, and populations not represented in training data. Grad-CAM is a post-hoc visualization and does not establish causal clinical reasoning. Softmax confidence is not calibrated. No threshold here is approved for screening or diagnosis. Before clinical research, obtain ethics/privacy approval, define intended use, lock the model and preprocessing versions, perform external validation, monitor drift, and provide a human override and incident-reporting process.
