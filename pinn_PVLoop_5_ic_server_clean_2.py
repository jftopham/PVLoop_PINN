import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from random import uniform
from scipy.interpolate import interp1d

# ----------------------------
# CONFIG
# ----------------------------
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch.manual_seed(123)

PATIENT_ID = 'patient_8_beat_5'
N_POINTS = 150
N_BEATS = 1
EPOCHS = 200
ITS = 1000

paths = {
    'volume': f'X:/JFT/PINNs/charite_matlab_examples/patients_data/{PATIENT_ID}.xlsx',
    'sample_ic': f'X:/JFT/PINNs/initial_data/initial_data_{PATIENT_ID}.xlsx',
    'output': 'X:/JFT/PINNs/data_output'
}

# ----------------------------
# UTILITIES
# ----------------------------

def interpolate_series(series, n_points=N_POINTS):
    return np.interp(np.linspace(0, n_points - 1, n_points), np.arange(series.size), series)

def cat_dict_tensors(tensor_dict, dim=1):
    return torch.cat(list(tensor_dict.values()), dim=dim)

def relu_constraint(x, lower=None, upper=None):
    penalty = 0.0
    if lower is not None:
        penalty += -((x - lower) - torch.abs(x - lower)) / 2
    if upper is not None:
        penalty += -((-x + upper) - torch.abs(-x + upper)) / 2
    return penalty

def compute_constraints(tau, sp, v0, vd, vm, sn, vmin):
    return (
        relu_constraint(tau, 30, 115) +
        relu_constraint(sp, 3, 80) +
        relu_constraint(v0, 8, 225) +
        relu_constraint(vd, 0.5, 65) +
        relu_constraint(vm, None, 380) +
        relu_constraint(vm, 1.1 * v0, None) +
        relu_constraint(v0 - vd, 5, None) +
        relu_constraint(sn, 1, 40) +
        relu_constraint(vmin - vd, 0.1 * vmin, None)
    )

def exact_solution_np(v, t, sn_0, splus_0, v0_0, vd_0, vm_0, p0_0, tau_0):
    v_np, t_np = v.to_numpy(), t.to_numpy()
    w = np.where(v_np <= v0_0,
                 sn_0 * np.log((v_np - vd_0) / (v0_0 - vd_0)),
                 -splus_0 * np.log((vm_0 - v_np) / (vm_0 - v0_0)))
    z = (p0_0 - w) * np.exp(-t_np / tau_0)
    return z + w

# ----------------------------
# MODEL
# ----------------------------

class FCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation=nn.Sigmoid):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), activation()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ----------------------------
# DATA LOADING
# ----------------------------

volume = pd.read_excel(paths['volume'])['vshort']
v_interpolated = interpolate_series(volume)

sample_ic = pd.read_excel(paths['sample_ic'])#.sample(n=1)

# ----------------------------
# SYNTHETIC DATA & TRAINING LOOP
# ----------------------------

for i in range(1): #sample_ic.shape[0]
    ic = sample_ic.iloc[i]
    tau_0, splus_0, v0_0, vd_0, vm_0, p0_0 = ic[['tau_base', 'splus_base', 'v0_base', 'vd_base', 'vm_base', 'p0_base']]
    sn_0 = splus_0 * (v0_0 - vd_0) / (vm_0 - v0_0)

    v = pd.Series(v_interpolated)
    t = pd.Series(np.linspace(0, ic['t_base'], N_POINTS))
    p = exact_solution_np(v, t, sn_0, splus_0, v0_0, vd_0, vm_0, p0_0, tau_0)

    # Prepare multi-beat synthetic data
    data = pd.DataFrame()
    ten = {}
    for j in range(N_BEATS):
        factor = 1 - 0.015 * j
        v_torch = torch.tensor(factor * v.values, dtype=torch.float32)
        t_torch = torch.tensor(t.values, dtype=torch.float32)
        p0_mod = torch.tensor((1 - 0.05 * j) * p0_0, dtype=torch.float32)

        def exact_solution(vv, tt):
            w = torch.where(vv <= v0_0,
                            sn_0 * torch.log((vv - vd_0) / (v0_0 - vd_0)),
                            -splus_0 * torch.log((vm_0 - vv) / (vm_0 - v0_0)))
            z = (p0_mod - w) * torch.exp(-tt / tau_0)
            return z + w

        p_torch = exact_solution(v_torch, t_torch)
        ten[f'v_{j+1}'] = v_torch.unsqueeze(-1)
        ten[f't_{j+1}'] = t_torch.unsqueeze(-1)
        ten[f'p_{j+1}'] = p_torch.unsqueeze(-1)

    # ------------------------
    # PINN Training
    # ------------------------

    pinn = FCN(N_BEATS, 1, hidden_dim=5, num_layers=2, activation=nn.Sigmoid)

    # Initialize parameters
    tau1 = torch.nn.Parameter(torch.tensor([uniform(35.0, 75.0)]))
    sp = torch.nn.Parameter(torch.tensor([uniform(5.0, 80.0)]))
    v0 = torch.nn.Parameter(ten['v_1'].mean() * uniform(0.1, 1))
    vd = torch.nn.Parameter(v0 / uniform(2, 5))
    vd.data = torch.clamp(vd.data, max=60)
    vm = torch.nn.Parameter(ten['v_1'].max() * uniform(1, 2))
    sn = lambda: sp * (v0 - vd) / (vm - v0)

    optimizer = torch.optim.Adam(list(pinn.parameters()) + [tau1, sp, v0, vd, vm], lr=1e-2)

    loss_history = {key: [] for key in ['tau', 'sp', 'v0', 'vd', 'vm', 'sn', 'loss_data', 'loss_phys', 'loss_constraint']}
    
    
    for step in range(EPOCHS * ITS + 1):
        inputs = cat_dict_tensors({f'v_{k+1}': ten[f'v_{k+1}'] for k in range(N_BEATS)})
        u_pred = pinn(inputs)

        # Data loss
        loss_data = sum(torch.mean((u_pred - ten[f'p_{j+1}'])**2) for j in range(N_BEATS))

        # Physics loss
        loss_phys = 0.0
        sn_val = sn()
        for j in range(N_BEATS):
            v = ten[f'v_{j+1}']
            t = ten[f't_{j+1}']
            p_target = ten[f'p_{j+1}']
            z_1 = torch.where(v <= v0, sn_val * torch.log((v - vd) / (v0 - vd)), torch.zeros_like(v))
            z_2 = torch.where(v > v0, -sp * torch.log((vm - v) / (vm - v0)), torch.zeros_like(v))
            w = (p_target[0] - z_1 - z_2) * torch.exp(-t / tau1)
            p_pred = w + z_1 + z_2
            loss_phys += torch.mean((p_pred - p_target) ** 2)

        # Constraints
        loss_constraint = compute_constraints(tau1, sp, v0, vd, vm, sn_val, ten['v_1'].min())

        # Final loss
        total_loss = loss_data + loss_phys + loss_constraint

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Record history
        loss_history['tau'].append(tau1.item())
        loss_history['sp'].append(sp.item())
        loss_history['v0'].append(v0.item())
        loss_history['vd'].append(vd.item())
        loss_history['vm'].append(vm.item())
        loss_history['sn'].append(sn_val.item())
        loss_history['loss_data'].append(loss_data.item())
        loss_history['loss_phys'].append(loss_phys.item())
        loss_history['loss_constraint'].append(loss_constraint.item())

    # ------------------------
    # Plot convergence
    # ------------------------

    plt.figure(figsize=(10, 5))
    plt.title(f"Convergence - Case {i}")
    for param in ['tau', 'sp', 'v0', 'vd', 'vm']:
        plt.plot(loss_history[param], label=f"{param} {loss_history[param][-1]:.2f} - {loss_history[param][-1]:.2f}")
    plt.xlabel("Training step")
    plt.ylabel("Value")
    plt.legend()
    # plt.legend([f'tau {tau} - {tau_0}',
    #             f'sp {sp} - {splus_0}',
    #             f'v0 {v0} - {v0_0}',
    #             f'vd {vd} - {vd_0}',
    #             f'vm {vm} - {vm_0}']) 
    #plt.grid(True)
    plt.show()
    
#%%
    # ----------------------------
    # Plot 1: Data Fidelity PV Loops
    # ----------------------------
    
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.set_title(f"Data Fidelity PV Loops - Case {i}")
    ax1.set_xlabel("Volume [ml]")
    ax1.set_ylabel("Pressure [mmHg]")
    ax1.grid(True)
    
    for j in range(N_BEATS):
        v = ten[f'v_{j+1}'].squeeze().numpy()
        p_data = ten[f'p_{j+1}'].squeeze().numpy()
        ax1.plot(v, p_data, label=f'Data Fidelity Beat {j+1}', linestyle='--')
        
    ax1.plot(v, p, label='Real PV Data', color='black', linewidth=2)
    
    ax1.legend()
    plt.tight_layout()
    plt.show()
    
    # ----------------------------
    # Plot 2: Final Parameters PV Loops
    # ----------------------------
    
    # Final values
    tau_final = torch.tensor(loss_history['tau'][-1])
    sp_final = torch.tensor(loss_history['sp'][-1])
    v0_final = torch.tensor(loss_history['v0'][-1])
    vd_final = torch.tensor(loss_history['vd'][-1])
    vm_final = torch.tensor(loss_history['vm'][-1])
    sn_final = torch.tensor(loss_history['sn'][-1])
    
    # Define model with final parameters
    def pv_model_final(v, t):
        w = torch.where(
            v <= v0_final,
            sn_final * torch.log((v - vd_final) / (v0_final - vd_final)),
            -sp_final * torch.log((vm_final - v) / (vm_final - v0_final))
        )
        z = (ten['p_1'][0] - w) * torch.exp(-t / tau_final)
        return z + w
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.set_title(f"Model PV Loops with Final Parameters - Case {i}")
    ax2.set_xlabel("Volume [ml]")
    ax2.set_ylabel("Pressure [mmHg]")
    ax2.grid(True)
    
    for j in range(N_BEATS):
        v = ten[f'v_{j+1}'].squeeze()
        t = ten[f't_{j+1}'].squeeze()
        p_model = pv_model_final(v, t).numpy()
        ax2.plot(v.numpy(), p_model, label=f'Model Beat {j+1}', linewidth=2)
    ax2.plot(v, p, label='Real PV Data', color='black', linewidth=2)
    ax2.legend()
    plt.tight_layout()
    plt.show()
    