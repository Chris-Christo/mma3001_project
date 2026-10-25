# MMA3001 Building Occupancy Forecasting

This university project investigates how well historical sensor readings can predict the number of people in a building space 30 minutes ahead. It compares a persistence baseline, linear regression and Support Vector Regression (SVR).

The engineering motivation is to provide information that could support future ventilation planning. The project does not control an HVAC system or demonstrate energy savings. The final modelling and evaluation cover one monitored space, `227f1068-9e41-469f-bef0-275f27a0bdba`.

## Main results

All three methods were evaluated on the same 2,260 test examples from January to March 2026.

| Method | MAE in people | RMSE in people |
| :--- | ---: | ---: |
| Persistence | 0.6819 | 1.0452 |
| Linear regression | 0.7013 | 0.9098 |
| SVR | 0.6273 | 0.8311 |

MAE is the average absolute prediction error. RMSE gives greater weight to larger errors. Lower values are better for both measures.

SVR was selected using validation results before the final test. Its test MAE was about 8.00% lower than persistence, although the absolute improvement was only about 0.055 people per forecast and it required more computation. This is a reduction in error, not a percentage accuracy score. The selected SVR settings were `C = 10`, `epsilon = 0.1` and `gamma = 0.01`.

The recorded values are in [comparison_metrics.csv](docs/results/227f1068/comparison_metrics.csv). Further results and limitations are in [the evaluation summary](docs/results/227f1068/final_evaluation_summary.md).

## Data and forecast setup

The project uses **Dataset 2: Monash Smart Infrastructure Occupancy and Environmental Data**, supplied for MMA3001. The original `DataSet2.zip` is not included in GitHub. Obtain it through the unit's teaching materials or teaching team with the required permission. Without this ZIP, the tests and saved results can still be reviewed, but the full analysis cannot be recreated.

October 2024 was used to compare the five available space identifiers. The final models use a longer period for the selected space:

| Dataset | Period | Complete examples |
| :--- | :--- | ---: |
| Training | March 2024 to August 2025 | 14,772 |
| Validation | September to December 2025 | 3,388 |
| Testing | January to March 2026 | 2,260 |

A forecast is scheduled every 15 minutes. The regression models use the current headcount, headcounts from 15 and 30 minutes earlier, time of day and day of week. Time and weekday are converted into numerical inputs in `src/features.py`. Persistence simply predicts that the current count will stay the same.

For each required headcount, preparation selects the latest reading at or before the requested time, provided it is no more than five minutes old. The same rule applies to the future target used for training and evaluation. An example is excluded if any required reading is unavailable; missing readings are not treated as zero.

Preparation stops if it finds invalid counts or conflicting counts at the same timestamp. The raw ZIP is never changed. Both regression models are fitted using training data only. Validation is used to choose settings, and the final test period is used for evaluation rather than further tuning.

## Repository layout

```text
notebooks/              Six notebooks, run in the order shown below
src/                    Shared Python functions used by the notebooks
tests/                  Automated checks of those functions
docs/html/              Generated source documentation
docs/results/           Small tables and figures from the original analysis
docs/reproducibility/    Records from the separate reproduction run
docs/                   Test explanations, reports and environment details
requirements.txt        Recorded Python package versions
LICENSE                 Assessment and review permissions
data/                   Raw and prepared data, created locally and ignored by Git
outputs/                Detailed results, created locally and ignored by Git
```

The written project report provides the full discussion and AI reflection and is submitted separately for assessment. The README explains how to run and review the computational work.

## Run the project

The recorded setup is **Windows 11 with Python 3.14.6**. Package versions are listed in [requirements.txt](requirements.txt), with the Python and system details in [docs/environment.txt](docs/environment.txt). Other systems have not been verified.

The instructions below use Windows PowerShell, Git and VS Code with the Python and Jupyter extensions. Run terminal commands from the project root, which is the folder containing `README.md` and `requirements.txt`.

### 1. Download a fresh copy

Start in a folder outside any existing copy of the project. Repository access is needed if it is private.

```powershell
git clone --branch main https://github.com/Chris-Christo/mma3001_project.git mma3001_review
cd mma3001_review
```

Open this new folder in VS Code. Use a fresh copy for a complete rerun rather than copying old prepared data or model outputs into it.

### 2. Create the Python environment

With Python 3.14 installed, run these commands individually:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

For the closest match to the recorded run, use Python 3.14.6. The last command should report `No broken requirements found.` If installation fails, check the message before changing any recorded package versions. The commands use the project's Python directly, so activating the environment is not required.

### 3. Add the original dataset

Create a `data` folder in the project root and **copy** the original ZIP into it:

```text
data/DataSet2.zip
```

Leave the ZIP unextracted and keep its name unchanged. The notebooks read the occupancy CSV from inside it. Do not copy an existing `data/prepared` folder or `outputs` folder from another run.

### 4. Run the implementation tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v --junitxml=docs/latest_test_results.xml
```

The supplied suite is expected to report **113 passed**. It checks error calculations, input preparation, model behaviour and evaluation safeguards using small examples. It does not need the building dataset and does not prove that the sensors count people accurately.

This command saves the current test report as `docs/latest_test_results.xml`, leaving the dated reproduction evidence unchanged. If a test fails, investigate it before running the analysis.

### 5. Run the notebooks in order

Select the `.venv` inside this project as the kernel for every notebook. For each notebook, restart the kernel, run all cells in order and save it before moving to the next one.

| Order | Notebook | Purpose |
| :--- | :--- | :--- |
| 1 | [1_occupancy_feasibility.ipynb](notebooks/1_occupancy_feasibility.ipynb) | Check the data and compare the monitored spaces. |
| 2 | [2_prepare_occupancy_data.ipynb](notebooks/2_prepare_occupancy_data.ipynb) | Prepare the forecast examples and chronological splits. |
| 3 | [3_persistence_baseline.ipynb](notebooks/3_persistence_baseline.ipynb) | Calculate the baseline errors. |
| 4 | [4_linear_regression.ipynb](notebooks/4_linear_regression.ipynb) | Train and evaluate linear regression. |
| 5 | [5_support_vector_regression.ipynb](notebooks/5_support_vector_regression.ipynb) | Compare twelve SVR settings using validation data. |
| 6 | [6_model_comparison.ipynb](notebooks/6_model_comparison.ipynb) | Compare the fixed methods on the final test period. |

The notebook working folder must be `notebooks`, because the code uses paths such as `../data/DataSet2.zip`. This differs from the project root used for terminal commands. Notebook 1 displays its working folder, which can help identify a path problem.

Notebook 2 creates `data/prepared/227f1068/`. Notebooks 3 to 6 create their results under `outputs/227f1068/`. Notebook 6 also updates the small summaries in `docs/results/227f1068/` within the copy being run.

## Reproduction and saved records

On 24 October 2026, I ran all six notebooks from a separate GitHub clone with a fresh Python environment on the same computer. All 113 tests passed. The saved example counts, MAE and RMSE matched the original nine model and period combinations, and the selected settings were unchanged.

The supporting records are in [docs/reproducibility/20261024](docs/reproducibility/20261024/). This was a repeat of an existing evaluation, not a new independent test or verification on different hardware. Runtime can vary between runs.

Some notebooks record file fingerprints to identify the exact code and data used. Even a comment change affects these fingerprints. If a source or protocol mismatch appears, do not manually edit or delete the old records to bypass it. Keep the original evidence and use a fresh clone with newly generated outputs. Since the test results are already known, a rerun should reproduce the existing method, not tune it against those results.

## Source documentation

The shared functions are documented in `docs/html/`. After downloading the repository, open the package page in a browser:

```powershell
Start-Process .\docs\html\src.html
```

To regenerate the pages from the current source files, run this from the project root:

```powershell
.\.venv\Scripts\python.exe -m pdoc src --docformat google -o docs/html
```

The pages describe the functions in `src`. Test explanations are in `docs/baseline_testing.md`, `docs/linear_testing.md`, `docs/svr_testing.md` and `docs/comparison_testing.md`.

## Main limitations

The models predict sensor reported counts for one monitored space. The readings were not checked against manual counts, and the exact room location was not confirmed. Only 26.16% of scheduled test forecasts had every required reading, so the errors do not describe performance at all times.

The accepted target can be up to five minutes earlier than the intended forecast time. Some regression predictions were negative and were kept unchanged in the error calculations. No energy savings, comfort improvements or suitability for controlling a real HVAC system were demonstrated.

## AI use and licence

ChatGPT assisted with the coding through out this project while all major decisions remained my own, for more detail review the Project_documentation.

The [Academic Assessment and Review Licence](LICENSE) permits review and reproduction for the purposes set out in that file. Other uses require permission as specified there. The Monash dataset, teaching materials and external libraries remain subject to their own terms.
