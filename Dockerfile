# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Make the start script executable (if using it directly)
# RUN chmod +x start.sh

# Expose the port that Streamlit runs on
EXPOSE 8501

# Run the Streamlit application
# CMD ["./start.sh"] # Option 1: Use the start.sh script
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

# Note: You will need to provide the necessary environment variables (PINECONE_API_KEY, GOOGLE_API_KEY)
# when running the Docker container, for example using the -e flag:
# docker run -p 8501:8501 -e PINECONE_API_KEY=your_key -e GOOGLE_API_KEY=your_key your_image_name