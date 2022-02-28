import fmpy
fmu = 'plant.fmu'
fmpy.dump(fmu)  # get information
res = fmpy.simulate_fmu(fmu)
