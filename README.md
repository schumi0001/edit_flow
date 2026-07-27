# WikiPulse

## Run

Requirements: Python 3.12, Java, and Docker Desktop.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd wikipulse

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m streamlit run dashboard/app.py

Open http://localhost:8501 and click Start Pipeline.

To stop, click Stop Pipeline, then press Control + C in the terminal.