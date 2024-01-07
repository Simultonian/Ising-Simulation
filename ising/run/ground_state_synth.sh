python groundstate/analytical/run_analytical.py --input data/input/groundstate.json
python -O groundstate/simulation/run_numerical.py --input data/input/groundstate.json
python plotter/groundstate/magnetization.py --input data/plotfig/groundstate.json
