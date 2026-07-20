# CuriosityPPOAgent — one-command entrypoints
#
# Usage:
#   make install          # install dependencies
#   make train-minigrid   # train MiniGrid DoorKey (potential-based)
#   make train-crafter    # train Crafter
#   make train-atari      # train Atari Montezuma (needs extra deps)
#   make eval             # evaluate a checkpoint
#   make test             # run unit tests
#   make demo             # start web demo
#   make plot-curves      # regenerate minigrid_curves.png
#   make help             # show all targets

.PHONY: install train-minigrid train-crafter train-atari eval test demo plot-curves help clean

PYTHON   ?= python
CONFIG   ?= experiments/minigrid_potential_shaping.yaml
CKPT     ?= results/checkpoints/minigrid/step_final.pt

# ── Install ────────────────────────────────────────────────
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "[ok] core dependencies installed"
	@echo "     For Atari support: $(MAKE) install-atari"

install-atari:
	$(PYTHON) -m pip install -r requirements_atari.txt

# ── Training ───────────────────────────────────────────────
train-minigrid:
	$(PYTHON) scripts/train_minigrid.py --config $(CONFIG)

train-crafter:
	$(PYTHON) scripts/train_crafter.py --config experiments/crafter_full.yaml

train-atari:
	$(PYTHON) scripts/train_atari.py --config experiments/atari_montezuma_full.yaml

# ── Evaluation ─────────────────────────────────────────────
eval:
	$(PYTHON) scripts/evaluate.py --checkpoint $(CKPT) --env minigrid --n-episodes 100

# ── Testing ────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v

test-coverage:
	$(PYTHON) -m pytest tests/ -v --cov=src/curiosity_ppo --cov-report=term-missing

# ── Web Demo ───────────────────────────────────────────────
demo:
	cd web && npm install && npm run dev

# ── Figures ────────────────────────────────────────────────
plot-curves:
	$(PYTHON) scripts/plot_minigrid_curves.py

# ── Clean artifacts ────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf *.egg-info dist build .coverage
	@echo "[ok] cleaned"

# ── Help ───────────────────────────────────────────────────
help:
	@echo "CuriosityPPOAgent Makefile targets:"
	@echo ""
	@echo "  make install          Install core Python dependencies"
	@echo "  make install-atari    Install Atari-specific dependencies (ROMs)"
	@echo "  make train-minigrid   Train MiniGrid DoorKey (potential-based)"
	@echo "  make train-crafter    Train Crafter"
	@echo "  make train-atari      Train Atari Montezuma's Revenge"
	@echo "  make eval             Evaluate a checkpoint (default: minigrid)"
	@echo "  make test             Run unit tests (144 tests)"
	@echo "  make test-coverage    Run tests with coverage report"
	@echo "  make demo             Start Vite+React web demo"
	@echo "  make plot-curves      Regenerate training curves PNG"
	@echo "  make clean            Remove cache / build artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON=$(PYTHON)"
	@echo "  CONFIG=$(CONFIG)"
	@echo "  CKPT=$(CKPT)"
