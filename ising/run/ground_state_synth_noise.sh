python groundstate/analytical/run_analytical.py --input data/input/groundstate_noisy.json
python -O groundstate/simulation/noise/run_numerical.py --input data/input/groundstate_noisy.json
python plotter/groundstate/magnetization_noise.py --input data/plotfig/groundstate_noisy.json
