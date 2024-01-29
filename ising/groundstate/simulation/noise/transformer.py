def transform_noisy_results(
    init_results: dict[int, dict[float, list[float]]], parameters: list[float]
) -> dict[float, dict[int, dict[float, float]]]:
    """
    Transforms the result dictionary from one form to the other.

    Initial form: dict[qubit, dict[h_value, list[answer]]]
    Here list[answers] is of the size parameters. Each answer corresponds to
    a parameter value for given qubit and h_value.

    Final form: dict[noise, dict[qubit, dict[h_value, answer]]]
    Here we have moved the noise to the top level for easier evaluation when
    plotting.
    """
    transformed_results = {}

    # Iterate over each qubit
    for qubit, h_value_dict in init_results.items():
        # For each h_value, create a dictionary mapping noise to answer
        for h_value, answers in h_value_dict.items():
            # Iterate over each noise level
            for i, noise in enumerate(parameters):
                # Associate noise with a dictionary mapping qubit to h_value to answer
                if noise not in transformed_results:
                    transformed_results[noise] = {}
                if qubit not in transformed_results[noise]:
                    transformed_results[noise][qubit] = {}
                transformed_results[noise][qubit][h_value] = answers[i]

    return transformed_results


