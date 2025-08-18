# ---- Build Stage ----
FROM python:3.11 AS builder

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libprotobuf-dev

# Install Rust
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Install Python build tools and all requirements
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir maturin setuptools wheel
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt


# ---- Final Stage ----
FROM python:3.11-slim

# Create a non-root user
RUN useradd --create-home appuser
USER appuser
WORKDIR /home/appuser/app

# Copy only the installed packages from the build stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy your application code
COPY . .

# Expose port and run the application
EXPOSE 8080
CMD ["python", "app.py"]