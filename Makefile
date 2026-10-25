# Task runner (GNU make). Windows: `winget install ezwinports.make`.
# Recipes are plain commands so they run under both cmd.exe (Windows) and sh (Linux/CI).

.DEFAULT_GOAL := help
export PYTHONUTF8 := 1

ifeq ($(OS),Windows_NT)
  BOOTSTRAP := py -3.11
  VENV_PY := .venv/Scripts/python.exe
else
  BOOTSTRAP := python3.11
  VENV_PY := .venv/bin/python
endif

PY ?= $(if $(wildcard $(VENV_PY)),$(VENV_PY),python)

.PHONY: help venv install lint format test data label parse-eval match-eval rank-eval fairness-eval eval report demo api ui

help:
	@echo Targets:
	@echo   venv     create .venv with Python 3.11
	@echo   install  install the package and dev tools
	@echo   lint     ruff check + format check
	@echo   format   auto-format and fix lint
	@echo   test     run pytest
	@echo   data     generate the SYNTHETIC resumes into data/
	@echo   label    hand-label the 50 sampled resumes in the terminal
	@echo   parse-eval  parse all resumes with Gemini and score the fields
	@echo   match-eval  title calls vs the role catalog, parse errors in matching
	@echo   rank-eval   ranking metrics, keyword/embedding/agent ablation, reason checks
	@echo   fairness-eval  score change when only the name or the college changes
	@echo   eval     all four evaluations, then report
	@echo   report   rewrite the README results from the results folder
	@echo   demo     rank job J1 in the terminal and review it
	@echo   api      HTTP API on port 8000, docs at /docs
	@echo   ui       Streamlit UI, start the api first

venv:
	$(BOOTSTRAP) -m venv .venv

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,app]"

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

test:
	$(PY) -m pytest

data:
	$(PY) -m shortlist.synthetic

label:
	$(PY) -m shortlist.label

parse-eval:
	$(PY) -m shortlist.eval_parse

match-eval:
	$(PY) -m shortlist.eval_match

rank-eval:
	$(PY) -m shortlist.eval_rank

fairness-eval:
	$(PY) -m shortlist.eval_fairness

eval: parse-eval match-eval rank-eval fairness-eval report

report:
	$(PY) -m shortlist.report

demo:
	$(PY) -m shortlist.cli run J1

api:
	$(PY) -m uvicorn shortlist.api:app --reload

ui:
	$(PY) -m streamlit run src/shortlist/ui.py
