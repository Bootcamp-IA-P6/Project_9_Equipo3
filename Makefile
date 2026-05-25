.PHONY: help install dev api ui build test clean docker docker-demo docker-clean stop

help:
	@echo "SignalMod — comandos disponibles:"
	@echo "  make install       Instala deps Python (uv) y frontend (pnpm)"
	@echo "  make dev           Arranca API (:8000) + UI (:5173) en paralelo"
	@echo "  make api           Solo la API FastAPI"
	@echo "  make ui            Solo el frontend React"
	@echo "  make build         Compila el frontend (frontend/dist)"
	@echo "  make test          Ejecuta pytest"
	@echo "  make docker        docker compose up --build (deja imágenes/volúmenes)"
	@echo "  make docker-demo   Demo efímera: arranca + al hacer Ctrl+C limpia todo"
	@echo "  make docker-clean  Limpia contenedores, imagen y volúmenes de SignalMod"
	@echo "  make stop          Mata procesos en :8000 y :5173"
	@echo "  make clean         Limpia .venv, node_modules y dist"

install:
	uv sync
	cd frontend && pnpm install

dev:
	@echo "→ API en http://localhost:8000   UI en http://localhost:5173"
	@trap 'kill 0' INT; \
		uv run uvicorn src.api.main:app --reload --port 8000 & \
		(cd frontend && pnpm run dev) & \
		wait

api:
	uv run uvicorn src.api.main:app --reload --port 8000

ui:
	cd frontend && pnpm run dev

build:
	cd frontend && pnpm run build

test:
	uv run pytest

docker:
	docker compose up --build

docker-demo:
	@echo "→ SignalMod en http://localhost:8000  (Ctrl+C para parar y limpiar)"
	@trap 'echo; echo "→ Limpiando contenedores, imagen y volúmenes..."; \
		docker compose down --rmi local --volumes --remove-orphans; \
		echo "✓ Todo limpio."' INT TERM EXIT; \
		docker compose up --build

docker-clean:
	@echo "→ Eliminando contenedores, imágenes y volúmenes de SignalMod..."
	-docker compose down --rmi local --volumes --remove-orphans
	@echo "✓ Limpieza completada."

stop:
	-@lsof -ti tcp:8000 | xargs kill -9 2>/dev/null || true
	-@lsof -ti tcp:5173 | xargs kill -9 2>/dev/null || true
	@echo "Procesos en :8000 y :5173 detenidos."

clean:
	rm -rf .venv frontend/node_modules frontend/dist
