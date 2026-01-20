import numpy as np
from pathlib import Path
from collections import deque
import random
from ase import units
from ase.build import add_vacuum

from ase.io import write
from ase.io import read

from ase.optimize import FIRE
from ase.md.nose_hoover_chain import MTKNPT
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

import re
from pathlib import Path

class MDMC:
    def __init__(
            self,
            calc, 
            temperature: float, 
            swaps: list, 
            seed=None, 
            logfile=None, 
            logsteps=100):
        
        # Check for restart and initialize atoms
        self.logfile = logfile
        self.logstep = logsteps
        self.temperature = temperature
        self._read_file()
        
        self.atoms.set_pbc((True,True,True))  
        self.atoms.calc = calc
        self.rng = np.random.default_rng(seed)

        
        self.beta = 1 / (8.617333262145e-5 * temperature)
        self.swaps = swaps

        directory = Path(".")
        pattern = re.compile(r"energies-300K-(\d+)")
        self.versions = [0]

        for file in directory.iterdir():
            if file.is_file():
                match = pattern.search(file.name)
                if match:
                    self.versions.append(int(match.group(1)))

    def _read_file(self):
        
        # Check Restart
        if Path(self.logfile).is_file():
            # Parse output
            self.restart = True
            with open(self.logfile) as f: # Output format is "STEP: ENERGY | (element1, element2) | successes"
                last_line = deque(f, maxlen=1)[0].split(":")
                self.initial = int(last_line[0]) + 1
                self.success = int(last_line[-1].split("|")[-1].strip())
            self.atoms=read(f'grx-mc-{self.temperature}K-{self.initial-1}.cfg')
        else:
            self.restart = False
            self.initial = 0
            self.success = 0

            self.atoms=read('grx.cfg')

    def _log_step(self, step, E_i, s1, s2):
        with open(file=self.logfile, mode="a") as myfile:
            myfile.write(f'{step}: {E_i:>12.6f} | {s1},{s2} | {self.success}\n')
        write(filename=f'grx-mc-{self.temperature}K-{step}.cfg', images=self.atoms, format="cfg")

    def create_sf(self, latparam, vacuum): 
        if not self.restart:
            _, _, zdim, *_ = self.atoms.cell.cellpar()
        
            z_displace = -1*latparam/np.sqrt(6)
            position = self.atoms.get_positions()
            mask = position[:,2] > ((zdim)/2)
            position[mask, 0] += z_displace
            self.atoms.set_positions(position)
            self.atoms.wrap()
            add_vacuum(atoms=self.atoms, vacuum=vacuum)

        tolerance = 5.0

        zs = [z for _, _, z in self.atoms.positions]
        min_z = min(zs)
        max_z = max(zs)

        self.z_upper = max_z - tolerance
        self.z_lower = min_z + tolerance

    def equilibriate(self, nsteps, md=True):
        if not self.restart:
            fmax=1e-4
            maxsteps=1000
            opt = FIRE(self.atoms)
            opt.run(fmax=fmax, steps=maxsteps)

            if md:
                nhalfsteps = int(nsteps/2)
                nhalfsteps = nsteps

                timestep = 1*units.fs
                MaxwellBoltzmannDistribution(atoms=self.atoms, temperature_K=self.temperature)
                
                #eq1 = NoseHooverChainNVT(
                #        self.atoms, 
                #        timestep, 
                #        temperature_K=self.temperature, 
                #        tdamp=100*units.fs)

                #eq1.run(nhalfsteps)

                # Pressure - 1 atm
                eq2 = MTKNPT(
                    self.atoms, 
                    timestep, 
                    temperature_K=self.temperature, 
                    pressure_au=6.3241E-07, 
                    tdamp=100 * units.fs,
                    pdamp=1000 * units.fs,
                    logfile=f"{self.temperature}.traj")
                
                eq2.run(nhalfsteps)

    def run(self, ncycles):
        # Initialize MD
        timestep = 1*units.fs
        MaxwellBoltzmannDistribution(
                atoms=self.atoms, 
                temperature_K=self.temperature)

        # Initialize MC
        Ei = self.atoms.get_potential_energy()
        energies = []
    
        # RUN MONTE CARLO SWAPS
        for step in range(self.initial, ncycles+self.initial):
            atoms_list = list(self.atoms.symbols)
            
            # Choose random indices of swap atoms
            while True:
                s1, s2 = self.swaps[self.rng.integers(len(self.swaps))] # Assign s1, s2 from swap combinations

                # First atom
                indices_1 = [i for i, v in enumerate(atoms_list) if v == s1]
                idx1 = (random.choice(indices_1) if indices_1 else -1)
                # Second atom
                indices_2 = [i for i, v in enumerate(atoms_list) if v == s2]
                idx2 = (random.choice(indices_2) if indices_2 else -1)

                # Exclude atoms at surface
                pos1 = self.atoms.positions[idx1, 2]
                pos2 = self.atoms.positions[idx2, 2]
                if not (self.z_lower < pos1 < self.z_upper):
                    continue
                if not (self.z_lower < pos2 < self.z_upper):
                    continue
                break
        
            # Attempt swap
            self.atoms.symbols[idx1] = s2
            self.atoms.symbols[idx2] = s1

            # Evaluate energies
            Ef = self.atoms.get_potential_energy()
            dE = Ef - Ei
            num = random.uniform(0, 1)
            prob = np.exp(-self.beta * dE)
            
            # Run MD
            if step % int(0.1 * len(self.atoms)) == 0:
                dyn = NoseHooverChainNVT(
                        self.atoms, 
                        timestep, 
                        temperature_K=self.temperature, 
                        tdamp=100*units.fs)
                dyn.run(10) #NVT
                Ei = self.atoms.get_potential_energy()

            # Accept
            if dE < 0 or num < prob:
                energies.append(Ef)
                Ei = Ef
                self.success += 1
            # Reject
            else:
                self.atoms.symbols[idx1] = s1
                self.atoms.symbols[idx2] = s2

            if step % self.logstep == 0 and step != 0:
                self._log_step(step, Ei, s1, s2) # Log step
                if not self.restart:
                    np.save(f"energies-{self.temperature}K.npy", energies)
                else:
                    np.save(f"energies-300K-{max(self.versions)+1}", energies)

