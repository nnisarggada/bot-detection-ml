#!/bin/bash

# Define the port numbers for each instance
ports=(5000 5001 5002 5003)

source env/bin/activate

# Start each instance of the Flask app
for port in "${ports[@]}"; do
  echo "Starting app.py on port $port"
  # Run the Flask app on the specified port and run it in the background
  python3 app.py --port=$port > flask_app_$port.log 2>&1 &
done

echo "All Flask instances are now running."
