#!/bin/bash
echo "Waiting for database start up"
sleep 5
python -m alembic upgrade head
python -m uvicorn preprocessor.main:app --app-dir src --env-file .env --host 0.0.0.0
