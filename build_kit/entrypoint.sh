#!/usr/bin/env bash

ollama serve &
time sleep 5  # give server time to start
uv run python -m app.main