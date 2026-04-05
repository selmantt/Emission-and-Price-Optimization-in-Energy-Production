# Emission and Price Optimization in Energy Production

A multi-objective optimization framework that balances **CO₂ emission reduction** and **electricity price minimization** in energy production using **Genetic Algorithm (GA)** and **Particle Swarm Optimization (PSO)**.

## Overview

Energy production requires balancing economic efficiency and environmental sustainability. This project formulates the problem as a constrained multi-objective optimization task and uses evolutionary algorithms to find optimal energy source mixes for **Germany (DE)** and **France (FR)**.

### Key Features (from `genetic.py`)

- **Dual Optimization:** GA (`POPULATION_SIZE=200`, `MAX_GENERATIONS=300`) and PSO (`NUM_PARTICLES=100`, `MAX_ITER=300`) working in parallel
- **Multi-Objective Fitness:** Weighted sum of normalized price (α=2.0), emissions (β=0.8), and renewable share (γ=0.1)
- **Realistic Constraints:** Max price increase ≤12%, emission reduction target ≥5%, per-source share change limits (±30% relative / ±1.5pp absolute)
- **30+ Energy Sources:** Solar, Wind, Nuclear, Natural Gas, Lignite, Hydro, Bioenergy, Geothermal, and more — each with source-specific operational and installation costs (€/kWh)
- **Country-Specific Cost Models:** Separate unit cost dictionaries for Germany (`unit_costs_euro_per_kwh_DE`) and France (`unit_costs_euro_per_kwh_FR`)
- **Amortized CAPEX Calculation:** Installation costs computed via `calculate_amortized_capex(capex_eur_per_kw, capacity_factor, lifetime_years)`

## Dataset

`combined_data_DE_FR_2020_2022.csv` — Combined energy production data from **Germany** and **France** (2020–2022), including:
- Energy source shares (%), renewable share, fossil energy share
- CO₂ emissions (tonnes), total production (GWh)
- Average household electricity price (€/kWh)

## Project Structure

```
├── utils/                                  # Utility functions and helpers
├── genetic.py                              # GA + PSO implementation with fitness functions
├── main.ipynb                              # Main notebook — analysis, visualization, results
├── combined_data_DE_FR_2020_2022.csv       # Dataset (Germany & France, 2020-2022)
└── README.md
```

## Usage

```bash
git clone https://github.com/selmantt/Emission-and-Price-Optimization-in-Energy-Production.git
cd Emission-and-Price-Optimization-in-Energy-Production

# Run optimization directly
python genetic.py

# Or explore via notebook
jupyter notebook main.ipynb
```

## Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `random` | Stochastic operations in GA/PSO |
| `matplotlib` | Visualization (in notebook) |

## Author

**Selman Turan Toker** — AI Engineering Student @ Istanbul Technical University
