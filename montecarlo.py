import sys
from itertools import combinations

from mclib import MDMC
from fairchem.core import FAIRChemCalculator, pretrained_mlip

"""
Usage: python3 montecarlo.py <int: nsteps> <bool: sf>
"""

# Specify simulation constants
try:
    temp = int(sys.argv[1])
except:
    temp = 300

seed = 123456

# Load MLIP
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", inference_settings="turbo", device="cuda", workers=4)
calc = FAIRChemCalculator(predictor, task_name="omat")

species = ['Ni','Co','Cr','Re','Ti','Al','W','Nb','C']
swaps = list(combinations(species, 2))

nsteps = int(sys.argv[2])
disl = sys.argv[3]

logfile = f"mc-sf-{temp}.txt"

# Initialize MC
mc = MDMC(calc=calc, 
        temperature=temp, 
        swaps=swaps, 
        seed=seed, 
        logfile=logfile, 
        young_modulus=188.2)

if disl:
    mc.create_sf(latparam=3.57)

mc.equilibriate(20000)
mc.run(nsteps)
