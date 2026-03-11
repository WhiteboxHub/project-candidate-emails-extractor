# Email Contact Extractor

**Automated vendor contact extraction from candidate email inboxes using AI-powered Named Entity Recognition (NER)**

A production-ready email processing service that automatically extracts recruiter and vendor contact information from candidate Gmail accounts, featuring GLiNER zero-shot NER, SpaCy entity recognition, and database-driven filtering for clean, accurate contact data.

---

## 🎯 Features

- **Multi-Method Extraction**: Regex → SpaCy → GLiNER fallback pipeline for maximum accuracy
- **Smart Filtering**: Database-driven junk/recruiter detection with BERT and LLM classifiers
- **Job Classification (Ollama/Groq)**: High-accuracy job validation using local or cloud LLMs
- **Zero-Shot NER**: GLiNER model extracts contacts without predefined patterns
- **Multiple Contact Sources**: Extracts from From, Reply-To, Sender, CC, and calendar invites
- **Duplicate Prevention**: Automatic deduplication by email and LinkedIn ID
- **UID Tracking**: Incremental processing - only processes new emails
- **Database Integration**: Saves to MySQL with activity logging
- **Test Mode**: Standalone testing without database (saves to JSON)

---

## 🏗️ Architecture

```
Candidate Email (IMAP) 
    ↓
Fetch Emails (batched)
    ↓
Filter (Junk/Recruiter Detection)
    ↓
Clean Text (HTML → Plain Text)
    ↓
Extract Contacts (Regex → SpaCy → GLiNER)
    ↓
Validate & Deduplicate
    ↓
Save to Database
    ↓
Log Activity
```

### Key Components

- **IMAP Connector**: Gmail authentication with app password support
- **Email Filter**: Database-driven keyword filtering + ML classification
- **Contact Extractor**: Multi-method NER with field-specific priority
- **UID Tracker**: JSON-based incremental processing tracker
- **API Client**: RESTful integration with Whitebox Learning Platform

---

## 📦 Installation

### Prerequisites

- Python 3.11+
- MySQL 8.0+
- Gmail account with [App Password](https://myaccount.google.com/apppasswords)

### Setup

1. **Clone Repository**
   ```bash
   cd project-candidate-emails-extractor
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

4. **Configure Environment**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your credentials:
   ```env
   # API Configuration
   API_BASE_URL=https://api.whitebox-learning.com
   API_EMAIL=your@email.com
   API_PASSWORD=your_password
   EMPLOYEE_ID=your_employee_id
   
   # Test Mode (optional)
   TEST_EMAIL=your.email@gmail.com
   TEST_APP_PASSWORD=your_16_char_app_password
   TEST_BATCH_SIZE=100
   ```

5. **Validate Setup**
   ```bash
   # validate_setup.py is currently unavailable
   ```

---

## 🚀 Usage

### 📧 Email Extraction Workflow

The main workflow for processing candidate inboxes end-to-end. It fetches emails, extracts recruiter/job info, and routes data based on validation results.

```bash
# Run the complete email extraction workflow
python src/run_workflow.py --workflow-key email_extractor
```

**Features:**
- **Incremental Processing**: Uses UID tracking to only process new messages.
- **Multi-Inbox Support**: Fetches candidate credentials from the database and processes all active inboxes.
- **NER-Based Routing**: 
  - ✅ **Finalized**: Jobs passing strict NER validation are sent to `/api/positions/` (Job Listings).
  - ⚠️ **Fallback**: Jobs failing NER but containing a valid email are sent to `/api/email-positions/bulk` (Email Positions).
- **Audit Logs**: Generates categorized JSON results in `output/extraction_results/` and detailed summary reports in `output/reports/`.

---

### 🤖 Job Classification & Validation

This service uses LLMs (Groq or Local Ollama) to classify raw job descriptions into `valid_job` or `junk` with high precision.

#### 1. Setup Infrastructure
The classifier works best with a local LLM to avoid rate limits and costs.

**Option A: Local Ollama (Recommended)**
1. Navigate to the LLM infrastructure directory:
   ```bash
   cd ../project-Ollama-local-llm
   ```
2. Start the Docker containers:
   ```bash
   docker-compose up -d
   ```
3. **Crucial**: Pull the required model inside the container:
   ```bash
   docker exec ollama ollama pull qwen2.5:1.5b
   ```
4. Verify the API is healthy:
   ```bash
   curl http://localhost:8000/health
   ```

**Option B: Groq Cloud (Fastest)**
Add your API key to `.env`:
```env
GROQ_API_KEY=your_gsk_key_here
MODEL_NAME=llama-3.1-8b-instant
```

#### 2. Run Classification
Go back to the main project and start the classification loop:

```bash
# Production Run
python llm_based_classifier.py --batch-size 20

# Dry Run (Test logic without saving to DB)
python llm_based_classifier.py --dry-run --batch-size 10
```

**Features:**
- **Persistent Progress**: Processes batches of raw records and marks them `parsed` in the database.
- **Auto-Retry**: Built-in retry logic with exponental backoff for API reliability.
- **Audit Logging**: Detailed reasoning for every classification saved to `classification_audit_llm.log`.
- **Normalization**: Automatically maps job types and work modes to valid database enums.

---
### Test Mode (No Database Required)

[Note: test_my_account.py is currently unavailable]

---

## ⚙️ Configuration

### `config/config.yaml`

Customize extraction behavior:

```yaml
extraction:
  enabled_methods:
    - regex      # Fast pattern matching
    - spacy      # NER entity extraction
    - gliner     # Zero-shot modern NER
  
  block_gmail: true
  extract_multiple_contacts: true
  
  email_priority:
    - reply-to
    - sender
    - from
    - cc
    - body

gliner:
  model: urchade/gliner_base
  threshold: 0.5
  entity_labels:
    - person name
    - company name
    - location
    - job title
```

**Filter Management:**  
All filtering rules (junk domains, recruiter keywords, patterns) are managed via `keywords.csv` and synchronized to the database. No hardcoded filters.

---

## 🧪 Testing & Validation

### Pre-Flight Check
[Note: validate_setup.py is currently unavailable]

### Diagnose Account
```bash
python diagnose_account.py
```

Troubleshoots IMAP connection issues.

---

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t email-extractor:latest .
```

### Run Service Mode
```bash
docker run --env-file .env \
  -e ENV_MODE=service \
  email-extractor:latest
```

### Run Test Mode
```bash
docker run --env-file .env \
  -e ENV_MODE=test \
  email-extractor:latest
```

---

## 📊 Database Schema

### Required Tables

**`candidate_marketing`**
- `id`, `email`, `imap_password`, `status`, `priority`

**`vendor_contact_extracts`**
- `name`, `email`, `phone`, `company`, `linkedin_id`, `location`, `extraction_source`

**`job_types`**
- Insert: `('bot_candidate_email_extractor', 'Extract vendor contacts from emails')`

**`job_activity_log`**
- `candidate_id`, `job_type_id`, `contacts_extracted`, `notes`, `timestamp`

---

## 🛠️ Utilities

| Script | Purpose |
|--------|---------|
| `service.py` | Main production service |
| `run_workflow.py` | Universal workflow runner (e.g., email_extractor) |
| `llm_based_classifier.py` | LLM-powered job classification engine |
| `diagnose_account.py` | IMAP connection troubleshooting |
| `sync_keywords_to_csv.py` | Sync database filters to CSV |
| `reset_tracker.py` | Reset UID tracker (reprocess emails) |

---

## 📝 Logs & Tracking

- **UID Tracking**: `last_run.json` (production), `last_run_test.json` (test)
- **Results**: `test_results_<timestamp>.json` (test mode)
- **Logging**: Console output with levels (INFO, WARNING, ERROR)

---

## 🔧 Troubleshooting

### Connection Failed

1. Enable [2FA](https://myaccount.google.com/security)
2. Generate [App Password](https://myaccount.google.com/apppasswords)
3. Use 16-character app password (not regular password)
4. Enable IMAP in Gmail Settings

### No Emails Fetched

- Check UID tracker: `last_run.json`
- Reset tracker: `python reset_tracker.py`
- Verify IMAP folder: Defaults to `INBOX`

### Database Errors

```bash
python validate_setup.py
```

Verify credentials in `.env` and MySQL server status.

---

## 📦 Dependencies

**Core:**
- `pyyaml`, `python-dotenv`, `fastapi`

**Email:**
- `email-validator`, `icalendar`

**NLP/Extraction:**
- `spacy`, `gliner`, `transformers`, `torch`

**Database:**
- `mysql-connector-python`

**ML:**
- `scikit-learn`, `numpy`, `pandas`

**See [`requirements.txt`](requirements.txt) for full list**

---

## 📄 License

Internal project for Whitebox Learning Platform

---

## 🤝 Contributing

For issues or improvements, contact the development team.

---

## 🔗 Quick Links

- [Gmail App Passwords](https://myaccount.google.com/apppasswords)
- [Enable 2FA](https://myaccount.google.com/security)
- [GLiNER Documentation](https://github.com/urchade/GLiNER)
- [SpaCy Models](https://spacy.io/models/en)
