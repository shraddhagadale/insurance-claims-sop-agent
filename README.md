# Insurance Claims SOP Agent

A hosted-demo-ready SOP harness for an insurance claims support agent. The agent follows a fixed workflow while still sounding like a realistic customer support representative:

```text
VERIFY_ID -> RESOLVE_INTENT -> PROCESS_CASE -> POST_PROCESS
```

The model can interpret language and phrase replies, but deterministic code owns identity verification, data access, phase transitions, consent, and output guarding.

## Requirements

- Python 3.13+
- An Anthropic API key

The demo requires a model token at startup. The browser never asks for the token; the backend reads it from server-side environment variables.

## Local Setup

Run these commands from the directory that contains the `insurance_claims/` folder:

```bash
python3 -m venv insurance_claims/.venv
source insurance_claims/.venv/bin/activate
pip install -r insurance_claims/requirements.txt
cp insurance_claims/.env.example insurance_claims/.env
```

Edit `insurance_claims/.env` and set:

```bash
ANTHROPIC_API_KEY=your_api_key_here
```

Run the app:

```bash
uvicorn insurance_claims.server:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000
```

## Docker

```bash
docker build -t insurance-claims-sop insurance_claims
docker run --env-file insurance_claims/.env -p 8000:8000 insurance-claims-sop
```

## Hosted Deployment

Recommended free option: Render Web Service on the Free compute plan.

Render setup:

1. Push this project to a Git repository.
2. In Render, create a new Web Service.
3. Select Docker as the runtime.
4. Set the Dockerfile path to `insurance_claims/Dockerfile` if the repository root contains this folder.
5. Choose the Free compute plan.
6. Add these environment variables in the Render dashboard:

```bash
ANTHROPIC_API_KEY=...
SOP_MODEL=claude-opus-5
SOP_AS_OF_DATE=2026-03-05
```

The Dockerfile binds to `${PORT:-8000}`, so it works locally and on hosts that inject a `PORT` environment variable.

## Demo Script

Paste this as the caller:

```text
I’m the policyholder. My name is Margaret Chen, policy POL-9921. I’m calling about my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.
```

Expected behavior:

- The agent verifies identity using at least 3 PII fields.
- It does not disclose claim details before verification.
- It remembers the denied-healthcare-January hint from the verification phase.
- It selects claim `CL-2048`.
- It answers from grounded claim/tool data.
- It offers an email summary and sends only after explicit consent.

## Tests

The tests use deterministic perception/template replies so they do not call the model.

```bash
insurance_claims/.venv/bin/python -m pytest insurance_claims/tests
```
