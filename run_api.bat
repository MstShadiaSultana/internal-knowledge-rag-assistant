@echo off
call .venv\Scripts\activate
uvicorn api:app --reload
