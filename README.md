# 1D-CNN-for-ECG-Classification
Using 1D CNN (convolutional neural network) deep learning technique to classify ECG (electrocardiography) signals as normal or abnormal. Trained with MIT-BIH Arrhythmia Database: https://www.physionet.org/physiobank/database/mitdb/  

## Web app

The repository includes a Flask landing page for ECG classification. A user can choose a demo ECG window
or upload a CSV file, then receive:

- classification: `אק״ג תקין` or `אק״ג לא תקין`
- probability that the signal is abnormal
- confidence level for the chosen class
- ECG signal preview chart

Run locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Public deployment

GitHub stores the code. To make the Flask app available from any computer, deploy the repository to Render.

Render settings:

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- Python runtime: `3.11.9` from `runtime.txt` / `render.yaml`

Recommended flow:

```bash
git add app.py templates/index.html static/styles.css README.md
git commit -m "Add ECG classification landing page"
git push
```

Then connect the GitHub repository in Render as a Python web service. Render can also use the included
`render.yaml`.

The full `ECG_window_df.csv` is intentionally ignored because it is too large for GitHub. The app includes `ECG_window_sample.csv` for public demo samples, and CSV upload still works for compatible ECG windows.

The model normalizes each ECG window with per-window min-max scaling before prediction:

```text
(x - window_min) / (window_max - window_min)
```
