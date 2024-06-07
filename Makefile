.PHONY: docs

BLUE=\033[0;34m
GREEN='\033[0;32m'
NC='\033[0m'

test:
	@poetry run pytest tests/

BUILD_PRINT = \e[1;34mBuilding $<\e[0m

format:
	@echo -e "\n${BLUE}Running Black against source and test files...${NC}\n"
	@black .

lint:
	@echo -e "\n${BLUE}Running Black against source and test files...${NC}\n"
	@black . --check
	@echo -e "\n${BLUE}Running Pylint against source...${NC}\n"
	@pylint feareu/**
	@pylint FEA/**
	@echo -e"\n${BLUE}Running Pylint against tests...${NC}\n"
	@pylint -d invalid-name tests/**
	@echo -e "\n${BLUE}Running Flake8 against source and test files...${NC}\n"
	@python -m flake8p
	@echo -e "\n${BLUE}Running Bandit against source files...${NC}\n"
	@bandit -r . -c "pyproject.toml"
	@echo -e "\n${BLUE}Running poetry check against pyproject.toml...${NC}\n"
	@poetry check

docs:
	sphinx-build -M html docs _build

clean:
	@find -type f -name coverage.xml -delete
	@find -type f -name .coverage -delete
	@find -type d -name .pytest_cache -exec rm -rf {} +

veryclean: clean
	@find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
