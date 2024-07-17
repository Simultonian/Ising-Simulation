import json
import os

def _truncate(num, digits=4):
    a, dot, b = str(num).partition(".")
    return a + dot + b[:digits]

def _get_data(file_name):
    if not os.path.exists(file_name):
        # If it doesn't exist, create a new file with an empty JSON object
        with open(file_name, 'w') as file:
            json.dump({}, file)
        print(f"{file_name} created.")
        return {} 

    else:
        with open(file_name, 'r') as file:
            results = json.load(file)
        return results

class PauliCounter:
    def __init__(self, system: str, num_qubits: int, time: float):
        """
        A class abstraction for getting the number of gate counts for the
        Pauli operation gate count. This is to avoid recomputation of the same
        value across multiple runs.
        """
        self.num_qubits = num_qubits
        time_str = _truncate(time)
        self.control_file_name = f"data/paulidata/control/{system}_{num_qubits}_{time_str}.json"
        all_data = _get_data(self.control_file_name)
        self.control_data = all_data.get("individual", {})
        self.control_total = all_data.get("total", {})

    def set_control_data(self, data):
        print(f"Saving data to {self.control_file_name}")
        with open(self.control_file_name, 'w') as file:
            json.dump(data, file)
