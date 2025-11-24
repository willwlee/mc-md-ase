import numpy as np
from collections import defaultdict
from pathlib import Path
from collections import deque

from ase import Atoms
from ase import units
from ase.build import bulk
from ase.cell import Cell

from ase.io import write
from ase.io import read

from ase.optimize import FIRE
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from ase.build import add_vacuum

class MDMC:
    def __init__(
            self,
            calc, 
            temperature: float, 
            swaps: list, 
            seed=None, 
            logfile=None, 
            logsteps=100,
            young_modulus=180):

        """
        Hybrid Molecular Dynamics / Monte Carlo (MD-MC) Simulation Engine for ASE.

        Parameters
        calc : ase.Calculator
            Calculator used to compute potential energies and forces.
        
        temperature : float
            Simulation temperature in Kelvin.
        
        swaps : list of tuple(str, str)
            List of allowed element swap pairs, e.g. `[("Ni","Co"), ("Ni","Cr")]`.

        seed : random seed, optional
            Random seed.  
            If None, a new default generator is created.  
            If an integer is passed, it is used as the seed.

        logfile : str or None, optional
            Path to the MC log file.  
            If the file already exists, the simulation attempts to restart from 
            the last logged step and corresponding saved CFG structure.

        logsteps : int, optional
            Frequency (in MC steps) at which energy, swaps, and acceptance
            statistics are written to the logfile. Default is 100.

        young_modulus : float, optional
            Young's modulus in GPa for estimating bulk modulus and barostat 
            compressibility when running NPT MD. Default is 180 GPa.
        """

        # Check for restart and initialize atoms
        self.logfile = logfile
        self.logstep = logsteps
        self.temperature = temperature
        self.read_file()

        self.atoms.set_pbc((True,True,True))  
        self.atoms.calc = calc
        self.rng = np.random.default_rng(seed)

        self.beta = 1 / (8.617333262145e-5 * temperature)
        self.swaps = swaps

        # Young's Modulus (GPa) - 188.2 for GRX
        p = 0.234 # Poisson Ratio
        K = young_modulus / (3 * (1-2*p)) # Bulk Modulus (GPa)
        self.compressibility = 1.0 / K # Compressibility (1 / GPa)

    def create_sf(self, latparam):
        
        _, _, zdim, *_ = self.atoms.cell.cellpar()
        z_displace = -1*latparam/np.sqrt(6)
        position = self.atoms.get_positions()
        mask = position[:,2] > ((zdim)/2)
        position[mask, 0] += z_displace
        self.atoms.set_positions(position)
        self.atoms.wrap()

        add_vacuum(self.atoms, vacuum=10.0)

    def read_file(self):
        
        # Check Restart
        if Path(self.logfile).is_file():
            # Parse output
            self.restart = True
            with open(self.logfile) as f: # Output format is "STEP: ENERGY | (element1, element2) | successes"
                last_line = deque(f, maxlen=1)[0].split(":")
                self.initial = int(last_line[0])
                self.success = int(last_line[-1].split("|")[-1].strip())
            
            self.atoms=read(f'grx-mc-{self.temperature}-{self.initial}.cfg')
        else:
            self.restart = False
            self.initial = 0
            self.success = 0

            self.atoms=read('grx.cfg')

    def _get_indices(self):
        idx = defaultdict(list)
        for i, sym in enumerate(self.atoms.get_chemical_symbols()):
            idx[sym].append(i)
        return dict(idx)

    def run(self, ncycles):

        # Initialize MD
        timestep = 1*units.fs
        MaxwellBoltzmannDistribution(atoms=self.atoms, temperature_K=self.temperature)
        dyn = NVTBerendsen(
            atoms=self.atoms, 
            timestep=timestep, 
            temperature_K=self.temperature,
            taut=1000 * units.fs)
        
        # Initialize MC
        E_i = self.atoms.get_potential_energy()
        idx = self._get_indices()
        cycle = int(0.1 * len(self.atoms))

        for step in range(self.initial, ncycles+self.initial):
            # Choose random indices of swap atoms

            s1, s2 = self.swaps[self.rng.integers(len(self.swaps))]
            i = self.rng.integers(len(idx[s1]))
            j = self.rng.integers(len(idx[s2]))
            idx_i, idx_j = idx[s1][i], idx[s2][j]

            # Propose swap
            atom_i, atom_j = self.atoms[idx_i], self.atoms[idx_j]
            atom_i.symbol, atom_j.symbol = s2, s1

            E_f = self.atoms.get_potential_energy()
            dE = E_f - E_i

            # Accept
            if dE < 0.0 or self.rng.random() < np.exp(-self.beta * dE):
                E_i = E_f
                self.success += 1
                idx = self._get_indices()
            # Reject
            else:
                atom_i.symbol, atom_j.symbol = s1, s2
            
            if step % self.logstep == 0 and step != 0:
                with open(file=self.logfile, mode="a") as myfile:
                    myfile.write(f'{step}: {E_i:>12.6f} | {s1},{s2} | {self.success}\n')
                write(filename=f'grx-mc-{self.temperature}K-{step}.cfg', images=self.atoms, format="cfg")

            # Run MD every cycle
            if step % cycle == 0:
                dyn.run(10)
                E_i = self.atoms.get_potential_energy()
                idx = self._get_indices()
    
    def equilibriate(self, nsteps, md=True):
        if not self.restart:
            fmax=1e-4
            maxsteps=1000
            opt = FIRE(self.atoms)
            opt.run(fmax=fmax, steps=maxsteps)

            if md:
                timestep = 1*units.fs
                MaxwellBoltzmannDistribution(atoms=self.atoms, temperature_K=self.temperature)
                
                eq_nvt = NVTBerendsen(
                    atoms=self.atoms,
                    timestep=timestep,
                    temperature_K=self.temperature,
                    taut=1000 * units.fs,
                    logfile=f"{self.temperature}.traj")
                eq_nvt.run(nsteps)
                
                eq = NPTBerendsen(
                    atoms=self.atoms, 
                    timestep=timestep, 
                    temperature_K=self.temperature, 
                    pressure_au=0.001 * units.bar, 
                    compressibility_au=(self.compressibility / units.GPa),
                    taup=1000 * units.fs,
                    taut=100 * units.fs,
                    logfile=f"{self.temperature}.traj")
                eq.run(nsteps)
                





