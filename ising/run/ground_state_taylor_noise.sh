python groundstate/analytical/run_analytical.py --input data/input/noisy_groundstate_taylor.json
python groundstate/simulation/noise/run_taylor.py --input data/input/noisy_groundstate_taylor.json
python plotter/groundstate/magnetization_noise.py --input data/plotfig/noisy_groundstate_taylor.json
