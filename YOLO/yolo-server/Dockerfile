# Use Python 3.10 base image
FROM python:3.10

# Set working directory
WORKDIR /code

# Copy requirements and install Python dependencies
COPY ./requirements.txt /code/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy application code
COPY ./app/src/ /code/app

# Set /code/app as the working directory
WORKDIR /code/app


# Run the application
CMD ["python", "server.py"]
