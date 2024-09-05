# Bot Detection Using Logistic Regression Model

This project involves training and using a logistic regression model to predict visitor types based on various features. The setup process includes creating a virtual environment, installing dependencies, and running the model script.

## Setup Instructions

Follow these steps to set up and run the project:

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/nnisarggada/bot-detection-ml
cd bot-detection-ml
```

### 2. Create and Activate a Virtual Environment

Create a virtual environment using the following command:

```bash
python -m venv env
```

Activate the virtual environment:

- **On macOS and Linux:**

  ```bash
  source env/bin/activate
  ```

- **On Windows:**

  ```bash
  env\Scripts\activate
  ```

### 3. Install Dependencies

Install the required Python packages by running:

```bash
pip install -r requirements.txt
```

### 4. Run the Setup Script

Run the setup script to ensure everything is configured properly:

```bash
python setup.py
```

### 5. Run the Logistic Regression Script

Execute the logistic regression script to train the model and make predictions:

```bash
python logistic_regression.py
```

## Usage

When running `logistic_regression.py`, you will be prompted to enter various features for the visitor. The script will then predict whether the visitor is a "Bot" or "Human" based on the provided input.
