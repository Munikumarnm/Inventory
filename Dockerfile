FROM python:3.9-slim

# Set working dir
WORKDIR /app

# Install system deps (if needed) and pip dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . /app

# Port configured by Cloud Run (use 8080)
ENV PORT 8080
EXPOSE 8080

# Run the Streamlit app in headless mode
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0", "--server.headless", "true"]
