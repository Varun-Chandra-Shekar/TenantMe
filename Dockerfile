# Base image: official slim Python 3.11 on Debian
FROM python:3.11-slim

# All subsequent commands run from /app inside the container
WORKDIR /app

# System libraries needed by Python packages (psycopg compiles against libpq)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — this layer is cached, so code edits don't
# trigger a full reinstall of deps every build
COPY requirements.txt pyproject.toml ./
COPY src ./src
COPY static ./static

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install -e .

# Copy the rest of the project
COPY . .

# Document which port the container listens on
EXPOSE 8000

# Run the API server when the container starts
CMD ["uvicorn", "tenantmate.app:app", "--host", "0.0.0.0", "--port", "8000"]