# Emission and Price Optimization in Energy Production

A multi-objective optimization framework that balances **CO₂ emission reduction** and **electricity price minimization** in energy production using **Genetic Algorithm (GA)** and **Particle Swarm Optimization (PSO)**.

## Overview

Energy production requires balancing economic efficiency and environmental sustainability. This project formulates the problem as a constrained multi-objective optimization task and uses evolutionary algorithms to find optimal energy source mixes for **Germany (DE)** and **France (FR)**.

## Key Features

- **Dual Optimization:** GA (`POPULATION_SIZE=200`, `MAX_GENERATIONS=300`) and PSO (`NUM_PARTICLES=100`, `MAX_ITER=300`) running in parallel
- **Multi-Objective Fitness:** Weighted sum of normalized price (α=2.0), emissions (β=0.8), and renewable share (γ=0.1)
- **Realistic Constraints:** Max price increase ≤12%, emission reduction target ≥5%, per-source share change limits (±30% relative / ±1.5pp absolute)
- **30+ Energy Sources:** Solar, Wind, Nuclear, Natural Gas, Lignite, Hydro, Bioenergy, Geothermal, and more — each with source-specific operational and installation costs (€/kWh)
- **Country-Specific Cost Models:** Separate unit cost dictionaries for Germany (`unit_costs_euro_per_kwh_DE`) and France (`unit_costs_euro_per_kwh_FR`)
- **Amortized CAPEX Calculation:** Installation costs computed via `calculate_amortized_capex(capex_eur_per_kw, capacity_factor, lifetime_years)`

## Dataset

`combined_data_DE_FR_2020_2022.csv` — Combined energy production data from Germany and France (2020–2022), including:

- Energy source shares (%), renewable share, fossil energy share
- CO₂ emissions (tonnes), total production (GWh)
- Average household electricity price (€/kWh)

**Sources:**
- Eurostat: [Energy Balances](https://ec.europa.eu/eurostat/databrowser/view/nrg_bal_peh/default/table) · [Electricity Prices](https://ec.europa.eu/eurostat/databrowser/view/nrg_pc_204/default/table)

-├── utils/                                  # Utility functions and helpers
-├── genetic.py                              # GA + PSO implementation with fitness functions
-├── main.ipynb                              # Main notebook — analysis, visualization, results
-├── combined_data_DE_FR_2020_2022.csv       # Dataset (Germany & France, 2020–2022)
-├── Report.pdf                              # Full project report
-└── README.md


## Usage
```bash
# Install dependencies (if any)
pip install numpy pandas matplotlib

# Run the main notebook
jupyter notebook main.ipynb
```

You can also run `genetic.py` directly to execute the optimization loops independently of the notebook.

## License

MIT
- EEA: [SIEC Vocabulary](https://dd.eionet.europa.eu/vocabulary/eurostat/siec/view)

## Project Structure
