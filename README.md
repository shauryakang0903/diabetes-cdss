# AI-Driven Explainable Clinical Decision Support System for Diabetes Diagnosis and Management

An academic Streamlit prototype for **physician-diagnosed diabetes classification**, using a frozen NHANES XGBoost model. It does not predict future incident diabetes.

## What the app does

- Accepts the fixed Core-10 inputs: age, sex, race/ethnicity, BMI, waist circumference, mean systolic/diastolic blood pressure, HDL, total cholesterol and smoking status.
- Produces the frozen model probability, development-derived LOW/MODERATE/HIGH strata and local SHAP explanations.
- Presents guideline-informed follow-up, monitoring and lifestyle prompts without prescribing medication.
- Reports input completeness and descriptive development-data applicability. Missing clinical measurements use the stored imputers; no fitting occurs in the app.
- Includes synthetic examples, model information and limitations. No datasets or training commands are needed to run it.

## Important disclaimer

**Academic prototype only: not a medical device, not clinically validated, not a diagnosis, and not a replacement for a qualified healthcare professional.** The target is physician-diagnosed diabetes status derived from DIQ010. Risk strata and operating thresholds are statistical development references, not clinically validated diagnostic cutoffs. SHAP describes model behavior, not causation. Recommendations are guideline-informed decision support, not an individualized treatment plan. The input-quality indicator is not a clinical confidence score or a probability that a prediction is correct.

Use synthetic cases for public demonstrations. The app requests no patient identifiers and does not save submitted records to disk; session results are held in memory. Do not enter identifiable patient information into this public research demonstration.

## Local run

Use **Python 3.13**, matching the frozen artifact's Python family. Install runtime dependencies in an environment separate from optional research packages:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts/check_streamlit_deployment.py
.venv\Scripts\python.exe -m streamlit run app.py
```

With an activated environment, Windows/generic launch commands are:

```text
python -m streamlit run app.py
streamlit run app.py
```

Linux uses the official same-version CPU-only XGBoost distribution to avoid CUDA/NCCL dependencies. Do not install both XGBoost distributions into one environment. The deployed app does not require pytest, LIME, PyTorch, TabM, notebooks, raw data or processed cohort CSVs.

## Deployment

1. Create an empty GitHub repository and push this project using the commands in [the deployment guide](docs/deployment/streamlit_cloud_deployment.md).
2. Open [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app**, and select your repository.
3. Set branch **main**, main file path **app.py**.
4. Open **Advanced settings** and explicitly select **Python 3.13**. Do not rely on the service's default Python version or an unsupported runtime.txt setting.
5. No secrets are required. Click **Deploy**, wait for installation, open the public URL and run the lower synthetic example. Expected probability is approximately **0.5871%** with LOW model-derived stratum.
6. Check that SHAP, recommendation tabs and Input Data Quality appear. On failure, open **Manage app → logs** and compare the installed versions and required files with the deployment guide.

The small model/metadata/rules files must be committed. Data, caches, experimental weights and generated research outputs are excluded by .gitignore; the optional saved global SHAP image is included. A missing optional image displays a message; missing model/metadata/rules prevent a successful assessment.

## Primary results

Previously recorded research results, not recomputed by the deployed app:

| Result | Value |
|---|---:|
| Primary XGBoost nested development ROC-AUC | 0.82658 |
| XGBoost temporal ROC-AUC | 0.81235 |
| XGBoost temporal Brier score | 0.10001 |
| Secondary compact TabM development ROC-AUC | 0.82862 |

TabM had no clear paired-bootstrap advantage over XGBoost and **is not the deployed model**. Temporal discrimination was weaker among participants with incomplete inputs. These findings do not establish clinical utility or individual prediction correctness.

Sources: the existing nested/temporal result tables, frozen metadata and [TabM methodology report](docs/validation/nhanes_tabm_deep_learning_experiment.md). Historical integration verification passed **207 tests before deployment integration**; this is not a claim about the deployment test count. Current verification is documented in [the deployment guide](docs/deployment/streamlit_cloud_deployment.md).

Deployment verification: **7 focused tests and all 225 normal-suite tests passed**, with no skips. A dataset-free clean runtime and both XGBoost package variants passed frozen synthetic inference; the real local Streamlit health endpoint returned HTTP 200. Actual Community Cloud deployment remains a manual step.

## Model provenance

- Development: NHANES 2011–March 2020, 25,580 adults.
- Separate completed temporal evaluation: NHANES 2021–2023, 7,536 adults.
- Artifact: models/nhanes/phase2_xgboost_pipeline.joblib (85,322 bytes; ordinary Git is sufficient).
- Numeric median imputation; categorical mode imputation and unknown-safe one-hot encoding. No app-time training, calibration or threshold selection.
- Frozen artifact SHA-256:

```text
fa2254598fc24d8c93bbf9cfe597aece0e1137f227098e58efe38bcf302a8528
```

The original metadata and input-reference JSON contain historical Windows provenance strings. They are preserved to retain their checksums; runtime paths are resolved from the source location and never use those recorded machine paths. Only load the trusted project artifact: a checksum verifies integrity, not the safety of arbitrary pickle files.

Research workflows require separate requirements files and local data. They are not executed during app startup. Do not regenerate the frozen model to fix a deployment error.
