# Deployment Instructions

To deploy the HR Attrition Rate application using Docker Compose, follow these steps:

1.  **Ensure Docker is running:**
    Make sure Docker Desktop or your Docker daemon is active on your system.

2.  **Navigate to the project root:**
    Open your terminal or command prompt and change the directory to the project's root:
    `C:\Users\shahu\OPEN CLASSROOMS\PROJET 5\HR Attrition Rate`

3.  **Build and run the services:**
    Execute the following command in your terminal:
    ```bash
    docker-compose up --build -d
    ```
    This command will perform the following actions:
    *   **Build Docker Images:** It will build the Docker images for both the FastAPI and Streamlit applications. This step is only performed if the images haven't been built before or if changes have been made to the Dockerfiles or their dependencies.
    *   **Start PostgreSQL Database:** It will start the PostgreSQL database service, which is defined in `docker-compose.yml`.
    *   **Start FastAPI Application:** It will start the FastAPI application service, making the API available.
    *   **Start Streamlit Application:** It will start the Streamlit application service, making the interactive dashboard available.
    *   The `-d` flag ensures that all containers run in **detached mode**, meaning they will run in the background, and your terminal will remain free for other commands.

4.  **Access the applications:**
    Once the services are up and running (this might take a few moments, especially on the first run), you can access the applications through your web browser:
    *   **FastAPI Documentation (Swagger UI):** Open your web browser and go to `http://localhost:8000/docs` to view the interactive API documentation, where you can test the API endpoints.
    *   **Streamlit Application:** Open your web browser and go to `http://localhost:8501` to access the Streamlit dashboard for employee attrition risk prediction.

5.  **To stop the applications:**
    When you are finished using the applications and want to stop the running Docker containers, execute the following command in your project's root directory:
    ```bash
    docker-compose down
    ```
    This command will stop and remove the containers, networks, and volumes created by `docker-compose up`.

You should now have both your FastAPI and Streamlit applications running locally via Docker Compose.