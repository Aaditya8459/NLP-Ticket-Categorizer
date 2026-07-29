# 🎫 Ticket Triage System

An intelligent, production-ready support ticket classifier that reads incoming tickets and automatically assigns them to the right team — with confidence-based human review fallback.

---

## 📁 Project Structure

```
ticket_triage_system/
├── config.yaml              ← 🔧 EDIT THIS: all tunable settings
├── train.py                 ← Train the model on your data
├── predict.py               ← Interactive CLI for single tickets
├── batch_predict.py         ← Process a CSV of tickets at once
├── requirements.txt         ← Python dependencies
├── data/
│   └── tickets.csv          ← 📥 PUT YOUR DATA HERE
├── models/                  ← Trained artifacts (auto-created)
│   ├── vectorizer.pkl
│   ├── classifier.pkl
│   └── label_encoder.pkl
├── reports/                 ← Evaluation reports (auto-created)
│   ├── evaluation.json
│   └── batch_predictions.json
├── src/
│   ├── config.py            # Config loader
│   ├── preprocessor.py      # Text cleaning & normalization
│   ├── features.py          # TF-IDF vectorization
│   ├── model.py             # Classifier training/evaluation
│   ├── triage_engine.py     # Main prediction orchestrator
│   └── utils.py             # Data loaders & helpers
└── notebooks/               # Your experiments go here
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd ticket_triage_system
pip install -r requirements.txt
```

### 2. Prepare Your Data

**Create a CSV file** at `data/tickets.csv` with exactly these columns:

```csv
subject,body,category
"Invoice overcharge","I was billed twice this month...","Billing"
"API 500 error","Production is down since 9am...","Technical"
"Leave request","I need 2 weeks sick leave...","HR"
"Dark mode suggestion","It would be great if...","General"
```

> **Supported formats:** `.csv`, `.json`, `.xlsx`  
> **Required columns:** `subject`, `body`, `category` (names are configurable in `config.yaml`)

### 3. Configure (Optional)

Open `config.yaml` and tweak:

```yaml
data:
  train_path: "data/tickets.csv"   # ← point to your file
  subject_col: "subject"           # ← change if your CSV uses different headers
  body_col: "body"
  label_col: "category"

triage:
  confidence_threshold: 0.60       # ← lower = more auto-assigned, higher = safer
```

### 4. Train the Model

```bash
python train.py
```

This will:
- Load and clean your data
- Convert text to TF-IDF features
- Train a Logistic Regression classifier
- Evaluate on a held-out test set
- Save artifacts to `models/`
- Write an evaluation report to `reports/evaluation.json`

### 5. Predict Live Tickets

**Interactive mode:**
```bash
python predict.py
```

Type a subject and body, see the category + confidence + routing instantly.  
Type `demo` to run 6 pre-loaded examples. Type `quit` to exit.

**Batch mode (CSV → JSON):**
```bash
python batch_predict.py --input data/new_tickets.csv --output reports/results.json
```

---

## ⚙️ How to Feed Your Own Data

### Option A: Replace the CSV (Easiest)

1. Delete or rename `data/tickets.csv`
2. Put your file in `data/` (e.g., `data/my_tickets.csv`)
3. Update `config.yaml`:
   ```yaml
   data:
     train_path: "data/my_tickets.csv"
   ```
4. Run `python train.py`

### Option B: Use Excel or JSON

Same as above — just change the path:
```yaml
data:
  train_path: "data/my_tickets.xlsx"
```
The loader auto-detects the format from the file extension.

### Option C: Different Column Names

If your file uses `title` instead of `subject` and `message` instead of `body`:
```yaml
data:
  train_path: "data/tickets.csv"
  subject_col: "title"
  body_col: "message"
  label_col: "team"
```

---

## 🧠 Architecture Decisions

| Choice | Why |
|--------|-----|
| **TF-IDF** | Downweights generic words, upweights discriminative terms (e.g., "invoice" → Billing). Fast and interpretable. |
| **Logistic Regression** | Linear boundaries work well for text. Calibrated probabilities = reliable confidence scores. Sub-millisecond inference. |
| **Naive Bayes fallback** | Fast probabilistic baseline. Good when you have very little data. |
| **Confidence threshold** | Tickets below 60% confidence are routed to human review instead of being misassigned. |
| **Rule-based priority** | Keyword scanning for "down", "critical", "harassment" flags urgent items instantly — no ML needed. |
| **No external NLP deps** | Built-in stopwords + simple stemmer mean zero setup friction. Optional NLTK Porter stemmer available. |

---

## 📊 Output Format

Each prediction returns:

```json
{
  "subject": "Production database is down",
  "body": "Our primary PostgreSQL instance went down...",
  "predicted_category": "Technical",
  "confidence": 0.87,
  "confidence_pct": "87.0%",
  "all_probabilities": {
    "Billing": 0.02,
    "General": 0.05,
    "HR": 0.06,
    "Technical": 0.87
  },
  "priority": "URGENT",
  "routing": "AUTO-ASSIGNED",
  "routing_reason": "Confidence 87.0% meets threshold",
  "cleaned_text": "product postgr instanc down custom servic offlin critic outag immedi escal sre team"
}
```

---

## 🔧 Advanced Tuning

### Improve accuracy
- **Add more data.** With <100 samples, confidence stays low. Aim for 500–2000+ tickets.
- **Switch to Porter stemmer** in `config.yaml`:
  ```yaml
  preprocessing:
    stemmer: "porter"
  ```
  (Requires `pip install nltk`)

### Change the model
```yaml
model:
  type: "naive_bayes"   # or "logistic_regression"
```

### Adjust human-review threshold
```yaml
triage:
  confidence_threshold: 0.75   # stricter = more human review
```

### Add custom stopwords
Create `data/my_stopwords.txt` (one word per line), then:
```yaml
preprocessing:
  stopwords_source: "data/my_stopwords.txt"
```

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: models/vectorizer.pkl` | Run `python train.py` first. |
| `Missing columns in data` | Check `config.yaml` — `subject_col`, `body_col`, `label_col` must match your CSV headers. |
| Very low confidence on everything | Normal with <50 training samples. Add more labeled data. |
| `ModuleNotFoundError: sklearn` | Run `pip install -r requirements.txt`. |

---

## 📜 License

Internal use only. Built for support queue automation.
