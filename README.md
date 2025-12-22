# PrivMap - Adaptive Differentially Private Spatial Analytics

A privacy-preserving platform for exploring sensitive spatial data with mathematical guarantees. PrivMap implements the **PrivTree** algorithm to generate adaptive heatmaps that provide high resolution in dense areas while protecting sparse regions with calibrated noise.

![PrivMap Demo](docs/demo.png)

## Overview

PrivMap allows data analysts to explore taxi pickup patterns in Porto, Portugal without compromising individual privacy. The system uses differential privacy - specifically the PrivTree hierarchical decomposition algorithm - to ensure that no individual's exact location or presence in the dataset can be identified.

### Key Features

- **Differential Privacy**: Mathematical guarantee that individual records cannot be identified
- **Adaptive Resolution**: Automatically adjusts detail based on data density using quadtree decomposition
- **Budget Tracking**: Monitor and manage privacy budget per session
- **Interactive Visualization**: Real-time Leaflet-based map with color-coded density cells
- **Consistent Counts**: Hierarchical structure ensures parent-child count consistency

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js)                       │
│  ┌───────────┐ ┌─────────────────┐ ┌─────────────────────────┐ │
│  │  Leaflet  │ │ Privacy Controls│ │   Budget Tracker        │ │
│  │    Map    │ │  (ε selector)   │ │   (Session Manager)     │ │
│  └───────────┘ └─────────────────┘ └─────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │ REST API
┌────────────────────────────▼────────────────────────────────────┐
│                         Backend (FastAPI)                        │
│  ┌───────────────────┐ ┌──────────────────┐ ┌────────────────┐ │
│  │   Privacy Guard   │ │   PrivTree Algo  │ │ Session Store  │ │
│  │  (Budget Deduct)  │ │   (Quadtree DP)  │ │ (PostgreSQL)   │ │
│  └───────────────────┘ └──────────────────┘ └────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      Data Layer (PostGIS)                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Porto Taxi Pickup Locations                    ││
│  │                 (1.7M trajectory points)                    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## The PrivTree Algorithm

PrivTree is a differentially private algorithm for hierarchical spatial decomposition that eliminates the dependency on a pre-defined maximum tree height. The key innovation is using a **biased count** with a decaying factor that allows constant noise regardless of tree depth.

### Algorithm Overview

1. Start with the entire spatial domain as the root node
2. For each node, compute a biased count: `b(v) = max{θ - δ, c(v) - depth(v) * δ}`
3. Add Laplace noise: `b̂(v) = b(v) + Lap(λ)`
4. Split if `b̂(v) > θ`, otherwise mark as leaf
5. Publish noisy counts only for leaf nodes

### Privacy Guarantee

PrivTree satisfies ε-differential privacy when:

- λ ≥ (2β - 1) / ((β - 1) \* ε)
- δ = λ \* ln(β)

Where β is the fanout (4 for quadtree) and ε is the privacy budget.

### Reference

This implementation is based on:

> Zhang, J., Xiao, X., & Xie, X. (2016). **PrivTree: A Differentially Private Algorithm for Hierarchical Decompositions**. SIGMOD '16. [arXiv:1601.03229](https://arxiv.org/abs/1601.03229)

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL 15+ with PostGIS (optional, demo works without it)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file and configure
cp env.example .env

# Run the server
python run.py
```

The API will be available at `http://localhost:8000`. See API docs at `http://localhost:8000/docs`.

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The app will be available at `http://localhost:3000`.

### Database Setup (Optional)

For production use with PostGIS:

```sql
CREATE DATABASE privmap;
\c privmap
CREATE EXTENSION postgis;
```

Update `DATABASE_URL` in `.env` and run the data ingestion script.

### Docker Setup (Recommended)

The easiest way to run PrivMap is using Docker Compose:

```bash
# Quick start - production mode
make prod

# Or manually:
docker compose up -d
```

This starts all services:

- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:8000
- **Database**: localhost:5432

#### Development with Docker

For development with hot-reload on the backend:

```bash
# Start database and backend with hot-reload
make dev

# Run frontend separately (for hot-reload)
cd frontend && npm run dev
```

#### Available Make Commands

| Command         | Description                            |
| --------------- | -------------------------------------- |
| `make dev`      | Start development environment          |
| `make prod`     | Start production environment           |
| `make build`    | Build all Docker images                |
| `make down`     | Stop all services                      |
| `make logs`     | View logs from all services            |
| `make clean`    | Remove containers, volumes, and images |
| `make test`     | Run backend tests                      |
| `make db-shell` | Open PostgreSQL shell                  |

## Usage

### Privacy Budget (ε)

The privacy parameter ε controls the privacy-utility tradeoff:

| Epsilon  | Privacy Level | Use Case                                  |
| -------- | ------------- | ----------------------------------------- |
| 0.01-0.1 | Very High     | Sensitive data, minimal disclosure        |
| 0.1-0.5  | High          | Strong protection with reasonable utility |
| 0.5-1.0  | Moderate      | Balanced approach                         |
| 1.0-2.0  | Low           | High utility, less privacy                |
| 2.0+     | Minimal       | Near-original data visibility             |

### Session Budget

Each session starts with a fixed privacy budget (default: ε = 5.0). Every query deducts from this budget. Once exhausted, start a new session to continue analyzing.

## Project Structure

```
privmap/
├── backend/
│   ├── app/
│   │   ├── privacy/           # PrivTree algorithm implementation
│   │   │   ├── laplace.py     # Laplace noise generation
│   │   │   └── privtree.py    # Core PrivTree algorithm
│   │   ├── routers/           # API endpoints
│   │   ├── services/          # Business logic
│   │   └── main.py            # FastAPI application
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/               # Next.js app router
│   │   ├── components/        # React components
│   │   ├── hooks/             # Custom hooks
│   │   └── lib/               # Utilities
│   └── package.json
└── README.md
```

## API Endpoints

### Session Management

- `POST /api/sessions/` - Create a new privacy session
- `GET /api/sessions/status` - Get current session status
- `DELETE /api/sessions/{token}` - Deactivate a session

### Spatial Queries

- `POST /api/spatial/decomposition` - Run PrivTree decomposition (with budget tracking)
- `GET /api/spatial/decomposition/quick` - Quick decomposition (no tracking)
- `GET /api/spatial/bounds` - Get default map bounds
- `GET /api/spatial/statistics` - Get dataset statistics

## Security Considerations

### Discrete Sampling

The Laplace mechanism implementation includes options for discrete sampling to mitigate floating-point precision attacks that could leak information through the exact values of noise samples.

### Budget Enforcement

Privacy budgets are enforced server-side. Clients cannot bypass budget limits by manipulating requests.

### No Raw Data Exposure

The system never exposes raw coordinate data. Only aggregate, noisy counts are returned.

## Future Work

- [ ] Support for 4D decomposition (origin-destination analysis)
- [ ] Sequence data privacy (trajectory modeling)
- [ ] Federated learning integration
- [ ] Extended range query support

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Porto Taxi Trajectory dataset from [Kaggle](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i)
- PrivTree algorithm by Zhang, Xiao, and Xie
- CARTO for dark map tiles
