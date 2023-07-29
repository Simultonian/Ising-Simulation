import json


def read_input_file(file_name):
    with open(file_name, "r") as file:
        data = json.load(file)

    INTEGERS = {"start_qubit", "end_qubit", "steps", "count_h", "count_time"}

    for key, value in data.items():
        if key in INTEGERS:
            data[key] = int(value)
        else:
            data[key] = float(value)
    return data
