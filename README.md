text
# Email Contact Extractor

Automated vendor contact extraction from candidate email inboxes with GLiNER-powered NER.

## Setup

### 1. Create Virtual Environment
python3 -m venv venv
source venv/bin/activate

-

### 2. Install Dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

-

### 3. Configure
cp config/.env.example config/.env
nano config/.env # Edit database credentials

text

### 4. Run
source venv/bin/activate
python service.py


## Configuration

Edit `config/config.yaml` to customize:
- Extraction methods (regex, spacy, gliner)
- Field-specific extractors
- Filtering rules
- Batch size

## Extraction Flow

Candidate Email → IMAP Connect → Fetch Emails → Filter (Junk/Recruiter)
→ Clean Text → Extract Contacts (Regex → Spacy → GLiNER)
→ Save to DB → Log Activity


**Virtual environment not activated:**
source venv/bin/activate



**Module not found:**
pip install -r requirements.txt



**Database connection failed:**
Check `.env` credentials and MySQL server status.
