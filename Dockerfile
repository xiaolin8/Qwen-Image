# =====================================================================================
# Stage 1: Base Image
#
# Using the official PyTorch image as requested for guaranteed compatibility.
# =====================================================================================
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

# =====================================================================================
# Stage 2: Setup Environment
# =====================================================================================
WORKDIR /app

# Set environment variables
# - PYTHONUNBUFFERED: Ensures Python output is sent straight to the terminal.
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# =====================================================================================
# Stage 3: Install Dependencies
#
# We copy only the requirements file first to leverage Docker's layer caching.
# The PyTorch base image already contains 'torch', pip will skip it.
# =====================================================================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =====================================================================================
# Stage 4: Copy Application Code and Set Entrypoint
# =====================================================================================
COPY . .

# Set the entrypoint to the python interpreter.
# This allows you to pass any python script and its arguments
# directly to the `docker run` command.
ENTRYPOINT ["python"]