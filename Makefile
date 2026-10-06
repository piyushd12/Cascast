# Cascadia — common tasks. Run `make help` for the list.
UV  := cd backend && uv run
BUN := cd frontend && bun

.PHONY: help setup grid data train evaluate export model test lint infra api simulator risk web up down

help:          ## Show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:         ## Install backend (incl. training deps) and frontend dependencies
	cd backend && uv sync --all-groups
	$(BUN) install

grid:          ## Export the IEEE 118-bus case + layout to backend/artifacts/grid.json
	$(UV) python -m cascadia.grid.case

data:          ## Simulate 100k labelled cascade scenarios
	$(UV) python -m cascadia.ml.datagen --n 100000

train:         ## Train CascadeGNN (uses CUDA when available)
	$(UV) python -m cascadia.ml.train --epochs 40

export:        ## Export the trained model to ONNX and verify parity
	$(UV) python -m cascadia.ml.export

evaluate:      ## Test-set metrics, benchmark and README figures
	$(UV) python -m cascadia.ml.evaluate

model: grid data train export evaluate  ## Full ML pipeline, end to end

test:          ## Run the backend test suite
	$(UV) pytest -q

lint:          ## Lint backend and frontend, type-check the frontend
	$(UV) ruff check src tests
	$(BUN) run lint && $(BUN) run typecheck

infra:         ## Start Redis and Postgres only (for local development)
	docker compose up -d redis postgres

api:           ## Run the API with auto-reload
	$(UV) uvicorn cascadia.api.main:app --reload --port 8000

simulator:     ## Run the grid digital twin
	$(UV) cascadia-simulator

risk:          ## Run a risk engine worker
	$(UV) cascadia-risk-engine

web:           ## Run the frontend dev server (http://localhost:5173)
	$(BUN) run dev

up:            ## Build and start the full stack (http://localhost:8080)
	docker compose up --build -d

down:          ## Stop the stack
	docker compose down
