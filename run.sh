#!/bin/bash
if [ -f ".venv/bin/streamlit" ]; then
    .venv/bin/streamlit run app.py
else
    streamlit run app.py
fi