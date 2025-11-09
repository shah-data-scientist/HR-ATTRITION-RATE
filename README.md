# HR Attrition Rate

## Project Overview and Goals

This project aims to analyze HR attrition data to identify key factors contributing to employee turnover. The ultimate goal is to build a predictive model that can help the HR department proactively address attrition risks.

## Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd hr-attrition-rate
    ```

2.  **Install dependencies:**
    This project uses Poetry for dependency management. Make sure you have Poetry installed.
    ```bash
    poetry install
    ```

3.  **Activate the virtual environment:**
    ```bash
    poetry shell
    ```

## Contribution Guidelines

-   **Branching:**
    -   `main`: This branch is for stable releases. Direct pushes are not allowed.
    -   `develop`: This is the main development branch. All feature branches should be created from `develop`.
    -   Feature branches: `feat/<feature-name>`
    -   Bugfix branches: `fix/<bug-name>`
-   **Commits:**
    -   Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.
-   **Pull Requests:**
    -   All changes must be submitted through a pull request.
    -   Pull requests should be made to the `develop` branch.
    -   Ensure your code is well-tested before submitting a pull request.

### CI/CD Pipeline

This project utilizes GitHub Actions to automate the Continuous Integration and Continuous Deployment process, ensuring code quality, facilitating testing, and enabling quick and reliable deployment.

**Pipeline Overview:**

*   **Triggers:** The pipeline is automatically triggered on `push` to the `main` branch and on `pull_request` events targeting the `main` branch.
*   **Jobs:**
    *   **`build-and-test`**: This job performs essential checks on the codebase:
        *   **Code Checkout**: Fetches the latest code from the repository.
        *   **Python Setup**: Configures the Python environment (version 3.13).
        *   **Dependency Installation**: installs project dependencies using Poetry.
        *   **Linting (Ruff)**: Checks for code style violations and potential errors.
        *   **Formatting Check (Black)**: Ensures code adheres to a consistent formatting style.
        *   **Unit Tests (Pytest)**: Executes all unit tests to verify code functionality.
    *   **`deploy`**: This job is responsible for deploying the application.
        *   **Trigger Condition**: Runs only after the `build-and-test` job successfully completes and only when changes are pushed to the `main` branch.
        *   **Deployment Target**: Currently configured as a placeholder for deployment to platforms like Hugging Face Spaces. Actual deployment steps would involve authentication and pushing application artifacts.

### Coding Standards

To maintain a consistent and high-quality codebase, the following coding standards are enforced:

*   **Linting**: `ruff` is used to identify and report problematic patterns in Python code.
    *   To run locally: `poetry run ruff check .`
*   **Formatting**: `black` is used to automatically format Python code to adhere to PEP 8.
    *   To run locally: `poetry run black .`
    *   To check formatting without applying changes: `poetry run black --check .`

### ML Experimentation Guidelines

Effective machine learning development requires structured experimentation. While specific tools for experiment tracking (e.g., MLflow, Weights & Biases) are not yet integrated, the following guidelines are recommended:

*   **Model Versioning**: Trained models (e.g., `outputs/employee_attrition_pipeline.pkl`) should be treated as artifacts. In a production setup, these would be versioned and stored in a model registry.
*   **Data Versioning**: For reproducibility, ensure that the datasets used for training and evaluation are versioned or clearly documented.
*   **Experiment Tracking**: For future enhancements, consider integrating tools to log parameters, metrics, and model artifacts for each experiment.
