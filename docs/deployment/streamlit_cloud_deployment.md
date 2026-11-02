# Streamlit Community Cloud deployment

## Audit and decisions

The pre-deployment root requirements.txt, .gitignore and README.md were empty. No existing Git repository, pyproject.toml or setup configuration was present. The main application already used source-relative pathlib paths and a cached, locked inference service; no redesign or prediction changes were necessary. Several research loaders/scripts used cwd-relative paths; these now resolve from their source file without changing cohort logic. No active Python source contained a literal user-specific Windows absolute path.

The source/runtime audit found no credential-like token or private-key patterns, .env or secrets.toml files. This is a pattern scan, not a guarantee about arbitrary future additions. Public NHANES datasets remain local and are excluded from Git. Generated provenance/results contain historical local paths; those outputs are excluded by default. The two frozen model/input-reference metadata JSON files retain inert Windows provenance strings because changing them would invalidate their checksums. They contain no detected credentials and are not used to locate runtime files. Do not perform blanket string replacements in frozen JSON.

## Runtime and Python

Select **Python 3.13** in Community Cloud's Advanced settings. Local serialization used Python 3.13.4. Same-major/minor Python and exact numerical package versions are preferred over moving to 3.11/3.12. Community Cloud currently supports security-maintained Python releases and permits version selection; its documented default is 3.12, so selection is important. No runtime.txt is added because that is not the documented Community Cloud selector.

| Runtime dependency | Version |
|---|---|
| Streamlit | 1.66.0 |
| NumPy | 2.5.3 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| XGBoost API/native version | 3.4.1 |
| joblib | 1.6.0 |
| SciPy | 1.18.1 |
| SHAP | 0.54.1 |
| Matplotlib | 3.11.2 |
| Pillow | 12.3.0 |

Root requirements.txt is inference-only. LIME is not imported by app.py's runtime dependency tree; it stays in its separate research requirements file. PyTorch, TabM, pytest and notebook packages are excluded. Research helper imports currently resolve to included lightweight source modules and do not load datasets or run experiments.

The standard XGBoost 3.4.1 Linux wheel declares nvidia-nccl-cu13. Linux therefore installs the official **xgboost-cpu==3.4.1** distribution instead (5.8 MB wheel); Windows/macOS retain xgboost==3.4.1. The two Linux wheels' 53 Python source files were byte-identical. A distribution-name fallback in the environment-version check recognizes xgboost-cpu; prediction, preprocessing, parameter checks and SHAP algorithms are unchanged. Never co-install both distributions. The frozen classifier already specifies device=cpu.

Linux CPython 3.13 binary wheels were verified for all pinned compiled runtime dependencies. Same-version joblib/sklearn/native XGBoost compatibility across operating systems cannot be guaranteed solely by matching versions; the synthetic inference and SHAP health check must pass after deployment. The original model was neither retrained nor resaved.

## Essential deployment files

- app.py, requirements.txt and .streamlit/config.toml.
- models/nhanes/phase2_xgboost_pipeline.joblib.
- models/nhanes/phase2_xgboost_metadata.json.
- models/nhanes/phase2_input_reference_metadata.json.
- configs/nhanes_recommendation_rules.json.
- The src/ Python tree and scripts/demo_nhanes_cdss.py, including their imported source helpers.
- scripts/check_streamlit_deployment.py for manual/CI preflight.

The optional image results/nhanes/xai/figures/shap_global_core10_bar.png is allowlisted in .gitignore. If absent, app.py displays an informational fallback. Neither cohort CSVs, raw XPT files, full SHAP arrays, historical result tables, experimental TabM files nor tests are needed for inference. Frozen thresholds and prior validation summaries are already in metadata/the existing UI; no experiment runs on startup.

The model is exactly **85,322 bytes**, safely below GitHub's regular-file size limit. No Git LFS or external artifact download is needed. No system packages are added speculatively; wheels provide the runtime used here. Do not disable CORS/XSRF protections or commit secrets to resolve deployment problems.

## Local preflight

From the project root, in Python 3.13:

```powershell
python -m pip install -r requirements.txt
python scripts/check_streamlit_deployment.py
python -m streamlit run app.py
```

The check also works when invoked by absolute script path from another working directory. It checks required files, frozen hashes, metadata, rules, fitted model, local SHAP and applicability output, and prevents CSV reads during inference. Lower synthetic probability must match 0.005870712921023369 within absolute tolerance 1e-8, with LOW stratum. Missing required artifacts fail clearly; they are not silently treated as optional.

No secrets, environment variables, command-line model parameters, cohort preprocessing or generated files are required. Streamlit cache_resource stores only the model/explainer/static rules with a lock. Patient inputs/results remain session-local, not globally cached or written to disk.

## GitHub preparation

No remote push or public repository was created automatically. Create an **empty** GitHub repository manually (do not initialize it with a competing README). From the project root, review the candidate files before committing:

```powershell
git init
git add .
git status --short
git diff --cached --stat
git commit -m "Prepare diabetes CDSS for Streamlit deployment"
git branch -M main
git remote add origin https://github.com/<USERNAME>/<REPO>.git
git push -u origin main
```

If a repository/remote already exists, inspect git status and git remote -v and reuse it; do not reinitialize or overwrite its remote blindly. The initial audit found none. Authenticate using your normal GitHub workflow; never paste tokens into source files or commit commands. GitHub's web interface offers **New repository → name → Public → Create repository**.

Do not commit .venv/, .deployment*/ test environments/downloads/bundles, .test_tmp*/, Python/test caches, data/, experimental weights, notebooks, .env files, secret keys or .streamlit/secrets.toml. Original files remain on disk. Generated research outputs are ignored except the optional UI image. Public research code/docs can remain in the repository; do not run its data-dependent tests as a deployment prerequisite.

## Community Cloud steps

1. Push the reviewed repository to GitHub.
2. Open https://share.streamlit.io/ and sign in/connect GitHub.
3. Choose **Create app**, then **Yup, I have an app** if prompted.
4. Select the repository, branch **main**, entrypoint **app.py**.
5. Open **Advanced settings**, select **Python 3.13**, and save. No secrets are required.
6. Choose **Deploy** and watch dependency/build logs.
7. Open the public .streamlit.app URL after installation.
8. Run the lower synthetic example: approximately 0.5871%, LOW, valid SHAP, four recommendation tabs and 10/10 Input Data Quality.
9. Also test the higher example and a missing clinical measurement. The result must retain its limitations/disclaimers.
10. The /_stcore/health endpoint should return HTTP 200 / ok; a green endpoint alone does not prove model inference, so perform the synthetic assessment too.

For failures, open **Manage app → logs**. Check Python selection and installed versions for dependency/unpickle errors; verify filename case, the three committed model/metadata files and rules for FileNotFoundError. Never retrain or rewrite frozen artifacts as a deployment workaround. If a model checksum fails, restore the verified original. Missing optional global images are informational. If cloud resource constraints persist, inspect logs and measured package sizes before changing packages or model behavior.

## Integrity and evidence

Frozen artifact SHA-256, before = after:

```text
fa2254598fc24d8c93bbf9cfe597aece0e1137f227098e58efe38bcf302a8528
```

Frozen metadata SHA-256, before = after:

```text
57fa864c5e81d28120ab414f94b326d86977535a88c22e79f8ac474037be66d3
```

The original input-reference metadata and recommendation configuration are also hash-checked. Test evidence is saved locally under ignored results/deployment/. Actual GitHub publication and Community Cloud startup require the manual steps above; local preparation is not a claim that a public URL has already been deployed.

Sources: [Community Cloud deployment/Python selection](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [dependency configuration](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies), [official XGBoost CPU distribution](https://xgboost.readthedocs.io/en/stable/install.html), and existing frozen metadata/research reports.

## Completed local verification (2026-11-02)

Created: scripts/check_streamlit_deployment.py, tests/test_streamlit_deployment.py, and this guide.

Modified root/config files: requirements.txt, .gitignore, README.md, .streamlit/config.toml.

Modified source: src/models/nhanes_phase2_model.py (distribution-version lookup only) and src/data/load_nhanes.py (source-relative raw-data root only).

Modified research script paths only: build_nhanes_development_cohort.py, build_nhanes_validation_2021_2023.py, build_nhanes_cohort_audit.py, final_audit_nhanes_development.py, inspect_nhanes_2011_2012.py, inspect_nhanes_2013_2014.py, inspect_nhanes_2015_2016.py, inspect_nhanes_2021_2023.py, inspect_nhanes_bmx.py, inspect_nhanes_bpxo.py, inspect_nhanes_demo.py, inspect_nhanes_diq.py, inspect_nhanes_hdl.py, inspect_nhanes_paq.py, inspect_nhanes_smq.py and inspect_nhanes_tchol.py (all under scripts/). No research data-building runner was executed.

- Focused deployment tests: **7 passed** (23.20 seconds).
- Normal project suite: **225 passed**, **0 skipped** (156.57 seconds). Data/environment-specific integration tests were available locally, so no exclusion was needed.
- Fresh Python 3.13.4 environment installed only root runtime dependencies; pip check reported no broken requirements. The original research .venv was not replaced.
- Both standard XGBoost and the official CPU-only 3.4.1 build loaded the original joblib artifact successfully. CPU-only regression preserved every prior output field exactly for all four captured complete/missing/uncommon synthetic cases.
- A separate deployment copy contained no datasets, research results, tests or TabM source modules. PyTorch, TabM, pytest and LIME were absent from its clean environment. Lower/higher examples reproduced probabilities **0.005870712921023369 / 0.5252653956413269**, with LOW/HIGH strata; missing HDL plus BMI 58 reproduced **0.10987188667058945**. The actual SHAP chart, four recommendation tabs and Input Data Quality displayed. Missing optional global imagery produced the expected information message.
- A real temporary Streamlit server serving that copy returned **HTTP 200 / ok** and was stopped without touching the user's server.
- Git candidate audit included every required artifact and the optional image, while excluding data/environments/caches. Approximately **1.13 MB** across **162 files**; largest candidate was the **141,544-byte** optional global SHAP image. No GitHub file-size issue or LFS requirement.
- Frozen model, metadata, input-reference JSON and recommendation-config hashes were identical before/after. app.py and SHAP/recommendation/input-quality algorithms were left unchanged; only the model loader's distribution-version lookup was extended.
- No existing Git repository was initialized, staged, committed or pushed. An ignored temporary bare repository was used only to verify .gitignore candidates. No public app was deployed automatically.

**Linux verification limit:** no installed Linux subsystem/container was available for native execution. All pinned compiled runtime packages had Python 3.13 Linux wheels, and the CPU/full Linux XGBoost Python APIs were source-identical, but these checks are not a substitute for the final Community Cloud build and synthetic assessment. A Windows clean-environment success alone is not claimed as a successful Linux deployment.

Evidence files in ignored results/deployment/ include pytest_focused.xml, pytest_full.xml, clean_standard_health.json, clean_cpu_health.json, clean_ui_cpu.json, cpu_regression.json, http_health.json, clean_cpu_freeze.txt, git_candidate_audit.json, xgboost_linux_wheel_audit.json and final_integrity.json. The first standard-build UI probe used the wrong image-counter identifier; its chart-count observation is superseded by the explicit, passing CPU-build image-render assertions.
