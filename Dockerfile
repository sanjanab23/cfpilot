# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Prevent Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
# Prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED 1

# Copy the requirements file into the container at /app
COPY backend/requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
# curl is optional (used only for healthcheck) — install is non-fatal if network is unavailable
RUN apt-get update && apt-get install -y --no-install-recommends curl || true \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY backend /app/backend

# Copy the frontend code
COPY frontend /app/frontend

# Create a directory for the database volume
RUN mkdir -p /app/data

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Set the working directory to the backend folder so imports work correctly
WORKDIR /app/backend

# Define environment variable for DB path (can be overridden at runtime)
ENV DB_PATH=/app/data/cfpilot.db

# Create a non-root user and switch to it
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Health check so Docker/Azure can detect unhealthy containers
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run uvicorn when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
