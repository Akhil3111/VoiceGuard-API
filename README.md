# Voice Guard API

This project separates API concerns (Gateway) from Business Logic (Sanitization/Intelligence) to ensure modularity and testability.

## Structure

- `app/`: Core application code
  - `api/`: Gateway layer (Endpoints, Dependencies)
  - `services/`: Business logic (Audio processing, Feature extraction, Scoring)
  - `schemas/`: Data Transfer Objects
  - `utils/`: Utilities (Logger)
- `ml_assets/`: Model storage
- `tests/`: Automated tests

## Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```
