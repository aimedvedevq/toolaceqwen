PYTHON ?= python
CONDA_ENV ?= ai

.PHONY: train quantize eval bench report clean

# Full pipeline
all: train quantize eval bench report

# Step 1: SFT + GRPO
train:
	$(PYTHON) scripts/sft.py
	$(PYTHON) scripts/grpo.py

# Step 2: Quantize
quantize:
	$(PYTHON) scripts/quantize.py --model ./output_grpo/merged --method fp8 --output ./output_grpo/fp8
	$(PYTHON) scripts/quantize.py --model ./output_grpo/merged --method w4a16 --output ./output_grpo/w4a16

# Step 3: BFCL evaluation (all configs)
eval:
	$(PYTHON) scripts/eval.py --all

# Step 4: Latency benchmarks
bench:
	$(PYTHON) scripts/bench.py --suite

# Step 5: Execute notebook and render HTML
report:
	jupyter nbconvert --to notebook --execute report.ipynb --inplace
	jupyter nbconvert --to html report.ipynb --output report.html

# Serve model (default: FP8 dynamic)
serve:
	$(PYTHON) scripts/run_inference_vm.py

# Pin dependencies
lock:
	pip freeze > requirements.lock

clean:
	rm -f report.html
	rm -rf results/bench/*.json
