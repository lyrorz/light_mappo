import fmpy
fmu = 'plante.fmu'
fmpy.dump(fmu)  # get information
res = fmpy.simulate_fmu(fmu)
