These values in benchmarkData.csv are directly from profiled numbers after running the profiler
you can see them in torch_profiler section or optionally can print out them if they are not loading
in line 290 in backendTorch.py (logging.info(profiler.key_averages())) and it should dump everything
to the terminal. 