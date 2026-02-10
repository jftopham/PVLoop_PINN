# PVLoop_PINN
![Logo](PINN_definition.png)
This is a repository to extract constitutive parameters from Left Ventricle Pressure Volume loops using PINNs.
Given Left Ventricle Pressure Volume loop data during diastole (pressure, volume and time), the algorithm recovers constitutive parameters from passive an active pressure equations.

## Quickstart

Follow these steps to set up your conda environment and start using this repository.

### 1. Clone the Repository

```bash
git clone https://github.com/jftopham/PVloop_PINN.git
cd PVloop_PINN
```

### 2. Create the Conda Environment

If an `environment.yml` file is provided, run:

```bash
conda env create -f environment.yml
```

If no `environment.yml` exists, create a new environment (replace `myenv` with your preferred name and python version):

```bash
conda create -n myenv python=3.9
conda activate myenv
# Manually install required packages:
# conda install <package1> <package2> ...
```

### 3. Activate the Conda Environment

```bash
conda activate myenv
```

### 4. Start Using the Repository

Run the main script:

```bash
python pinn_PVloop.py
```

---

**Troubleshooting:**
- If you encounter issues, ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed.
- For more details, see the [official conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

---
  year={2025},
  url={https://github.com/jftopham/PVloop_PINN}
}
```
