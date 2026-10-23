# MMA3001 — Building Occupancy Forecasting

## Project Overview

- This project investigates whether historical occupancy sensor measurements can be used to predict the number of occupants in a building space 30 minutes into the future.
- The intended engineering application is building ventilation demand planning, where occupancy forecasts could inform future HVAC system operation. By predicting future room headcounts, a HVAC system could calculate and operate an appropriate ventilation plan ahead of time.
- The HVAC system would use more then just future headcount in performing this task however the scope of the project is simply on establishing aon algorithm to predict heacount 30mins into the future.

## Dataset

- The project uses Dataset 2: Monash Smart Infrastructure Occupancy and Environmental Data.
- The occupancy records contain timestamps, building-space identifiers, occupancy statuses and recorded headcounts.

## Current Progress

- Python development environment and GitHub repository established.
- Initial occupancy data exploration completed for October 2024.
- Five monitored building spaces compared.
- Data availability and observation gaps investigated.
- Potential 30-minute forecasting examples identified.
- Regression model development and validation are the next stages.

## Running the Feasibility Notebook

1. Clone the repository and open it in VS Code.
2. Create a Python virtual environment named `.venv`.
3. Install the required packages: `pandas`, `matplotlib` and `ipykernel`.
4. Place the original `DataSet2.zip` in the `data/` folder.
5. Open `notebooks/1_occupancy_feasibility.ipynb`.
6. Select the project's Python environment as the Jupyter kernel.
7. Execute the notebook cells in order.

The current notebook uses the relative data path `../data/DataSet2.zip`, assuming its working directory is `notebooks/`.

## Repository Structure

- `notebooks/` — Exploratory analysis and visualisations.
- `data/` — Local datasets, excluded from Git.
- `docs/` — Project documentation.
- `src/` — Reusable computational methods (planned).
- `tests/` — Implementation tests (planned).
- `README.md` — Project overview and usage instructions.

## Current Limitations

- Sensor reported headcounts have not been independently verified against actual people counts.
- Observations occur at irregular intervals, and the physical locations associated with some space identifiers remain uncertain.
- The final forecasting method, building-space selection and validation strategy are still being developed.