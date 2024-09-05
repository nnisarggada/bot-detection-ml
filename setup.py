import subprocess
import os

scripts = ["generate_data.py", "data_clean.py", "undersampling.py", "plot_data.py"]

def run_script(script_name):
    try:
        result = subprocess.run(['python', script_name], check=True, text=True, capture_output=True)
        print(f"Output of {script_name}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:\n{e.stderr}")

def main():
    for script in scripts:
        if os.path.isfile(script):
            print(f"Running {script}...")
            run_script(script)
        else:
            print(f"{script} does not exist.")

if __name__ == "__main__":
    main()
