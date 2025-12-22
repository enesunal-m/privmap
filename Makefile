# PrivMap Makefile
# Convenience commands for development and deployment

.PHONY: help dev prod build up down logs clean test ingest ingest-sample ingest-docker

# Default target
help:
	@echo "PrivMap - Differentially Private Spatial Analytics"
	@echo ""
	@echo "Usage:"
	@echo "  make dev         Start development environment (DB + backend with hot-reload)"
	@echo "  make prod        Start production environment (all services)"
	@echo "  make build       Build all Docker images"
	@echo "  make up          Start all services (production)"
	@echo "  make down        Stop all services"
	@echo "  make logs        View logs from all services"
	@echo "  make clean       Remove all containers, volumes, and images"
	@echo "  make test        Run backend tests"
	@echo "  make ingest      Import all taxi data to database"
	@echo "  make ingest-sample  Import 100k sample records"
	@echo "  make db          Start only the database"
	@echo "  make backend     Start only the backend (requires db)"
	@echo "  make frontend    Start only the frontend (requires backend)"
	@echo ""

# Development mode - backend with hot-reload, frontend runs separately
dev:
	docker compose -f docker-compose.dev.yml up -d
	@echo ""
	@echo "Development environment started!"
	@echo "  - Database: localhost:5432"
	@echo "  - Backend:  http://localhost:8000 (with hot-reload)"
	@echo ""
	@echo "Run frontend separately:"
	@echo "  cd frontend && npm run dev"

# Production mode - all services
prod: build
	docker compose up -d
	@echo ""
	@echo "Production environment started!"
	@echo "  - Frontend: http://localhost:3000"
	@echo "  - Backend:  http://localhost:8000"
	@echo "  - Database: localhost:5432"

# Build all images
build:
	docker compose build

# Start all services
up:
	docker compose up -d

# Stop all services
down:
	docker compose down
	docker compose -f docker-compose.dev.yml down

# View logs
logs:
	docker compose logs -f

# Clean up everything
clean:
	docker compose down -v --rmi all
	docker compose -f docker-compose.dev.yml down -v --rmi all
	@echo "All containers, volumes, and images removed"

# Run tests
test:
	cd backend && python -m pytest app/privacy/tests/ -v

# Start only database
db:
	docker compose -f docker-compose.dev.yml up -d db
	@echo "Database started at localhost:5432"

# Start backend only
backend:
	docker compose -f docker-compose.dev.yml up -d backend
	@echo "Backend started at http://localhost:8000"

# Start frontend in development mode (not in docker)
frontend:
	cd frontend && npm run dev

# Install dependencies
install:
	cd backend && pip install -r requirements.txt
	cd frontend && npm install

# Database shell
db-shell:
	docker exec -it privmap-db-dev psql -U privmap -d privmap

# Backend shell
backend-shell:
	docker exec -it privmap-backend-dev /bin/bash

# Data ingestion
ingest:
	@echo "Ingesting taxi data into database..."
	cd backend && python scripts/ingest_data.py --csv ../train.csv
	@echo "Done!"

ingest-sample:
	@echo "Ingesting sample data (100k records)..."
	cd backend && python scripts/ingest_data.py --csv ../train.csv --limit 100000
	@echo "Done!"

ingest-docker:
	@echo "Ingesting data via Docker..."
	docker exec privmap-backend-dev python scripts/ingest_data.py \
		--csv /app/data/train.csv \
		--database-url postgresql+asyncpg://privmap:privmap_secret@db:5432/privmap
	@echo "Done!"

