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

## Usage

### Running the Streamlit Application

To launch the interactive Streamlit application:

```bash
poetry run streamlit run app.py
```

This will open the application in your web browser, allowing you to upload data, adjust prediction thresholds, and view attrition risk predictions and SHAP explanations.

### Running the API

To start the FastAPI application:

```bash
poetry run uvicorn api.app.main:app --host 0.0.0.0 --port 8000
```

The API documentation will then be available at `http://localhost:8000/docs`.

### Training the Model

To retrain the attrition prediction model:

```bash
poetry run python train.py
```

This script will process the raw data, train the model, and save the updated model artifact and associated metadata in the `outputs/` directory.

## Deployment

This project can be deployed using Docker. Dockerfiles are provided for both the Streamlit application and the FastAPI.

### Docker Deployment (Example for Streamlit)

1.  **Build the Docker image:**
    ```bash
    docker build -f Dockerfile.streamlit -t hr-attrition-streamlit .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 8501:8501 hr-attrition-streamlit
    ```

Similar steps apply for `Dockerfile.api`.

## Authentication

This project currently does not implement explicit user authentication for the Streamlit UI or the FastAPI. Access control would typically be handled at the infrastructure level (e.g., VPN, API Gateway, internal network restrictions).

For production deployments, consider integrating an authentication layer (e.g., OAuth2, JWT) for the FastAPI and appropriate access management for the Streamlit application.

## Security

*   **Data Handling:** Ensure sensitive data is handled in accordance with organizational policies and regulations (e.g., GDPR, CCPA). Avoid storing personally identifiable information (PII) in plain text.
*   **Dependency Management:** Regularly update project dependencies to mitigate known vulnerabilities. Use tools like `poetry update` and security scanners.
*   **API Security:** If exposing the FastAPI to external networks, implement rate limiting, input validation, and secure communication (HTTPS).
*   **Environment Variables:** Sensitive configurations (e.g., database credentials, API keys) should be managed using environment variables and never hardcoded in the codebase.

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