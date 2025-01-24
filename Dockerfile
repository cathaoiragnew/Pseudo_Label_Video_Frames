# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Define environment variables (optional)
ENV PYTHONUNBUFFERED=1

# Run the Python app when the container starts
CMD ["python", "create_pseudo_data.py"]