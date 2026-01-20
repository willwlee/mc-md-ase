# MD+MC Hybrid Simulation Framework (ASE + MACE)

This repository contains a **hybrid Molecular Dynamics (MD) and Monte Carlo (MC)** workflow built on **ASE (Atomic Simulation Environment)**, with optional support for **MACE machine‑learning interatomic potentials**. The code is designed for high‑temperature alloy simulations, allowing **chemical species swaps** under thermodynamic control while maintaining atomic relaxation via MD.

The implementation is:
- Use **MD** to locally relax atomic structures and sample phase space
- Use **Monte Carlo swaps** to explore configurational (chemical) disorder
- Accept/reject swaps using the **Metropolis criterion** at fixed temperature

This framework is well suited for **complex concentrated alloys**, **high‑entropy alloys**, and simulations involving **defects or stacking faults**.

---

## Features

- Hybrid **MD–MC** algorithm
- Built on **ASE** (optimizers, MD integrators, IO)
- Supports **MACE** machine‑learning potentials (GPU‑enabled)
- Canonical (NVT) and isothermal–isobaric (NPT) equilibration
- Restartable simulations
- Species‑restricted Monte Carlo swaps
- Surface‑aware swaps (exclude top/bottom atomic layers)
- Automatic trajectory and energy logging
- SLURM batch script included

---

## Code Structure

```
.
├── mclib.py            # Core MDMC class (MD + MC logic)
├── montecarlo.py       # Main executable script
├── montecarlo.sbatch   # Example SLURM submission script
├── grx.cfg             # Initial atomic structure (ASE CFG format)
└── README.md
```

---

## Requirements

### Python Packages

- Python ≥ 3.9
- numpy
- ASE
- mace‑torch (for MACE potentials)

Install core dependencies:

```bash
pip install numpy ase
```

Install MACE (example):

```bash
pip install mace-torch
```

You will also need a trained **MACE model file** (e.g. `mace-mpa-0-medium.model`).

---

## Methodology

### 1. Molecular Dynamics (MD)

- Geometry optimization using **FIRE**
- Temperature initialization using **Maxwell–Boltzmann distribution**
- Equilibration via:
  - NPT ensemble (MTK Nose–Hoover chains)
  - Optional NVT Nose–Hoover chains during MC sampling

### 2. Monte Carlo (MC) Swaps

- Randomly select a **pair of chemical species**
- Randomly select atoms of each species
- Enforce **bulk‑only swaps** (exclude surface atoms via z‑coordinate filtering)
- Accept or reject swaps using:

\[ P = \min\left(1, e^{-\beta \Delta E} \right) \]

where:

\[ \beta = \frac{1}{k_B T} \]

---

## MDMC Class (`mclib.py`)

### Initialization

```python
mc = MDMC(
    calc=calculator,
    temperature=300,
    swaps=[('Ni','Co'), ('Cr','Re')],
    seed=123456,
    logfile="mc-sf-300.txt"
)
```

**Key arguments**:

| Argument | Description |
|--------|-------------|
| `calc` | ASE calculator (e.g. MACECalculator) |
| `temperature` | Simulation temperature (K) |
| `swaps` | List of allowed species swap pairs |
| `seed` | RNG seed |
| `logfile` | Monte Carlo log file |
| `logsteps` | Logging frequency |


### Restart Capability

If `logfile` exists, the simulation automatically:
- Reads the last completed MC step
- Reloads the last saved atomic configuration
- Continues the simulation seamlessly

---

### Surface / Stacking Fault Setup

```python
mc.create_sf(latparam=3.57, vacuum=10)
```

This:
- Applies a shear displacement consistent with a stacking fault
- Adds vacuum along the z‑direction
- Defines **swap‑eligible bulk region** by excluding atoms near surfaces

---

### Equilibration

```python
mc.equilibriate(nsteps=100, md=True)
```

Steps:
1. Geometry optimization (FIRE)
2. Thermalization
3. NPT equilibration at 1 atm

---

### Production Run

```python
mc.run(ncycles=10000)
```

During a production run:
- MC swaps attempted each cycle
- Short MD bursts applied periodically
- Energies logged
- Accepted configurations written to disk

---

## Main Script (`montecarlo.py`)

### Usage

```bash
python montecarlo.py <nsteps> [temperature] [sf]
```

Examples:

```bash
python montecarlo.py 5000
python montecarlo.py 20000 1200 True
```

### Script overview

- Loads a **MACECalculator** on GPU
- Defines multicomponent alloy species
- Automatically generates all pairwise swaps
- Controls stacking‑fault activation

---

## Output Files

| File | Description |
|-----|-------------|
| `mc-sf-300.txt` | MC log (step, energy, swap, success count) |
| `grx-mc-300K-*.cfg` | Atomic snapshots |
| `energies-300K.npy` | Accepted MC energies |
| `*.traj` | ASE trajectory files |

---

## SLURM Execution

Use `montecarlo.sbatch` as a template for HPC runs:

```bash
sbatch montecarlo.sbatch
```

Ensure CUDA, Python, and MACE environments are correctly loaded.

---

## Extending the Code

- Replace **MACE** with any ASE‑compatible calculator
- Modify swap constraints (e.g. nearest‑neighbor only)
- Add biasing potentials or umbrella sampling
- Extend logging for thermodynamic analysis

---

## Practical notes

- Use **small MD bursts** during MC to control computational cost
- Monitor acceptance ratios (stored in log file)
- For high temperatures, consider tighter MD damping constants
- Ensure sufficient vacuum for surface simulations

---

## Citation

If you use this framework in published work, please cite:

- ASE
- MACE
- Relevant MD–MC methodology papers

---

## License

This code is provided as‑is for research use. Users are encouraged to adapt it as needed and cite relevant sources.

