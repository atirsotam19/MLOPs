# Variables
PYTHON = python3
PIP = pip
REQ_FILE = requirements.txt
PIPELINE_SCRIPT = pipelines_registry.py

# Install dependencies from requirements.txt
install:
	$(PIP) install -r $(REQ_FILE)

# Run the pipelines by executing the pipeline registry script
run_pipelines:
	$(PYTHON) $(PIPELINE_SCRIPT)

# Install dependencies and then run pipelines
all: install run_pipelines

# Clean temporary Python files and cache
clean:
	rm -rf __pycache__
	rm -rf *.pyc
	rm -rf *.pyo

.PHONY: install run_pipelines all clean
