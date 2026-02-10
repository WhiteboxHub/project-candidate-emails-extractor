# Email Contact Extractor

**Automated vendor contact extraction from candidate email inboxes using AI-powered Named Entity Recognition (NER)**

A production-ready email processing service that automatically extracts recruiter and vendor contact information from candidate Gmail accounts, featuring GLiNER zero-shot NER, SpaCy entity recognition, and database-driven filtering for clean, accurate contact data.

---

## 🎯 Features

- **Multi-Method Extraction**: Regex → SpaCy → GLiNER fallback pipeline for maximum accuracy
- **Smart Filtering**: Database-driven junk/recruiter detection with ML classifier
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
   API_BASE_URL=https://whitebox-learning.com
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

### Production Mode (Database-Driven)

Run the service to process all candidates in the database:

```bash
python service.py
```

**What it does:**
- Fetches candidates from `candidate_marketing` table
- Connects to each candidate's Gmail via IMAP
- Processes emails incrementally (UID tracking)
- Saves extracted contacts to `vendor_contact_extracts`
- Logs activity to `job_activity_log`

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
