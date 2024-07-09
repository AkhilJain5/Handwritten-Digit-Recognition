# Use the official Python image from the Docker Hub
FROM python:3.8

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the working directory
COPY . .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for the Flask app
EXPOSE 5000

# Set the environment variable to ensure output is shown in the console
ENV FLASK_APP app.py

# Run the Flask app
CMD ["flask","run", "--host=0.0.0.0"]
