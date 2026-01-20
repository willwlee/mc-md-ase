import sys
from itertools import combinations

from mclib import MDMC
from mace.calculators import MACECalculator
calc = MACECalculator(model_path='mace-mpa-0-medium.model', device='cuda')

"""
Usage: python3 montecarlo.py <int: nsteps> <bool: sf>
"""

nsteps = int(sys.argv[1])
# Specify simulation constants
try:
    temp = int(sys.argv[2])
except:
    temp = 300
try:
    disl = sys.argv[3]
except:
    disl = True

seed = 123456

# Load potential

species = ['Ni','Co','Cr','Re','Ti','Al','W','Nb']
swaps = list(combinations(species, 2))

logfile = f"mc-sf-{temp}.txt"

# Initialize MC
mc = MDMC(calc=calc, 
        temperature=temp, 
        swaps=swaps, 
        seed=seed, 
        logfile=logfile)

if disl:
    mc.create_sf(latparam=3.57, vacuum=10)

mc.equilibriate(100)
mc.run(nsteps)
