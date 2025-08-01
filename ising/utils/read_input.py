import json


def read_input_file(file_name):
    with open(file_name, "r") as file:
        data = json.load(file)

    INTEGERS = {
        "start_qubit",
        "end_qubit",
        "steps",
        "count_h",
        "count_time",
        "qubit_count",
    }
    STRING = {"method", "noise"}
    ARRAYS = {"polarization"}

    for key, value in data.items():
        if key in INTEGERS:
            data[key] = int(value)
        elif key in STRING:
            continue
        elif key in ARRAYS:
            data[key] = [float(x) for x in value]
        else:
            data[key] = float(value)
    return data


def read_json(file_name):
    with open(file_name, "r") as file:
        json_str = file.read()
        data = json.loads(json_str)
    return data
