import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from random import uniform

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
data_folder = r'X:/JFT/PINNs/charite_matlab_examples/IVC_occ/patients_data_occ/github'

# ----------------------------
# File utilities
# ----------------------------
def get_excel_files(directory):
    """Return a list of Excel files in a directory."""
    return [f for f in os.listdir(directory) if f.lower().endswith(('.xlsx', '.xls'))]

# ----------------------------
# Concatenate tensors from dict
# ----------------------------
def cat_dict_tensors(tensor_dict, dim=1):
    return torch.cat(list(tensor_dict.values()), dim=dim)

# ----------------------------
# Fully Connected Network
# ----------------------------
class FCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation=nn.Sigmoid):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), activation()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ----------------------------
# Constraint functions
# ----------------------------
def relu_constraint(x, lower=None, upper=None):
    penalty = 0.0
    if lower is not None:
        penalty += -((x - lower) - torch.abs(x - lower)) / 2
    if upper is not None:
        penalty += -((-x + upper) - torch.abs(-x + upper)) / 2
    return penalty    

def compute_constraints(tau, sp, v0, vd, vm, sn, data_vmin):
    return (
        relu_constraint(tau, 30, 115) +
        relu_constraint(sp, 3, 80) +
        relu_constraint(v0, 8, 225) +
        relu_constraint(vd, 0.5, 65) +
        relu_constraint(vm, 1.1 * v0, 380) +
        relu_constraint(sn, 1, 40) +
        relu_constraint(v0, vd + 5) +
        relu_constraint(0.9 * data_vmin, vd) # vd < 0.9*ESV
    )
files = get_excel_files(data_folder)
data = pd.read_excel(os.path.join(data_folder, files[0]))
epochs=100
its=1000
patience=5000
delta=1e-2
# ----------------------------
# Training function
# ----------------------------
#def train_pinn(data, epochs=100, its=1000, patience=5000, delta=1e-2):
N_BEATS = data.shape[1] // 3
ten = { 
        f't_{i}': torch.tensor(data[f't_{i}']*1000).unsqueeze(-1)
        for i in range(N_BEATS)
}
ten.update({
        f'v_{i}': torch.tensor(data[f'vshort_{i}']).unsqueeze(-1)
        for i in range(N_BEATS)
})
ten.update({
        f'p_{i}': torch.tensor(data[f'pshort_{i}']).unsqueeze(-1)
        for i in range(N_BEATS)
})

pinn = FCN(N_BEATS, 1, 5, 2)
torch.manual_seed(123)

    # Learnable parameters
v0_init = ten['v_0'].mean().item()
tau = nn.Parameter(torch.tensor(uniform(35.0, 75.0)))
sp  = nn.Parameter(torch.tensor(uniform(5.0, 80.0)))
v0  = nn.Parameter(torch.tensor(v0_init * uniform(0.1, 1)))
vd  = nn.Parameter(torch.tensor(min(v0.item()/uniform(2, 5), 60.0)))
vm  = nn.Parameter(torch.tensor(ten['v_1'].max().item() * uniform(1.1, 2)))
sn  = lambda: sp * (v0 - vd) / (vm - v0)

optimizer = torch.optim.Adam(list(pinn.parameters()) + [tau, sp, v0, vd, vm], lr=1e-3)

loss_history = {k: [] for k in ['tau','sp','v0','vd','vm','sn','loss_data','loss_phys','loss_constraint','times','iterations']}
inputs = cat_dict_tensors({f'v_{k}': ten[f'v_{k}'].float() for k in range(N_BEATS)})

start = time.time()
best_loss, counter = float('inf'), 0

for step in range(epochs * its + 1):
        u_pred = pinn(inputs)

        # Data loss
        loss_data = sum(torch.mean((u_pred - ten[f'p_{j}'])**2) for j in range(N_BEATS))

        # Physics loss
        sn_val, loss_phys = sn(), 0.0
        for j in range(N_BEATS):
            v, t, p_target = ten[f'v_{j}'], ten[f't_{j}'], ten[f'p_{j}']
            pp = torch.where(v <= v0,
                             sn_val * torch.log((v - vd) / (v0 - vd)),
                             -sp * torch.log((vm - v) / (vm - v0)))
            p_pred = (p_target[0] - pp) * torch.exp(-t / tau) + pp
            loss_phys += torch.mean((p_pred - p_target) ** 2)

        # Constraints
        loss_constraint = compute_constraints(tau, sp, v0, vd, vm, sn_val, ten[f'v_{N_BEATS-1}'].min())
        
        # if step % 1000 == 0: # check data fidelity
        #     print(u_pred[0:5])

        total_loss = loss_data + loss_phys + loss_constraint

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Early stopping
        current_loss = tau.item()+sp.item()+v0.item()+vd.item()+vm.item()
        if step > patience:
            if abs(best_loss - current_loss) > delta:
                best_loss, counter = current_loss, 0
            else:
                counter += 1
            if counter >= patience:
                break

        # Logging
        for k,v in zip(['tau','sp','v0','vd','vm','sn'], [tau, sp, v0, vd, vm, sn_val]):
            loss_history[k].append(v.item())
        loss_history['loss_data'].append(loss_data.item())
        loss_history['loss_phys'].append(loss_phys.item())
        loss_history['loss_constraint'].append(loss_constraint.item())

loss_history['times'].append(time.time()-start)
loss_history['iterations'].append(step)
#    return loss_history, u_pred, ten, N_BEATS

# ----------------------------
# Example run
# ----------------------------
data_folder = r'X:/JFT/PINNs/charite_matlab_examples/IVC_occ/patients_data_occ/github'
files = get_excel_files(data_folder)
data = pd.read_excel(os.path.join(data_folder, files[0]))

#loss_history, u_pred, ten, N_BEATS = train_pinn(data)


#%% PLOTS

# ----------------------------
# Plot 1: Convergence
# ----------------------------
    
plt.figure(figsize=(10, 5))
plt.title("Convergence - Case")
for param in ['tau', 'sp', 'v0', 'vd', 'vm']:
    plt.plot(loss_history[param], label=f"{param} {loss_history[param][-1]:.2f}")
plt.xlabel("Training step")
plt.ylabel("Value")
plt.legend()
plt.show()

# ----------------------------
# Plot 2: Data Fidelity PV Loops
# ----------------------------

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.set_title("Data Fidelity PV Loops - Case")
ax1.set_xlabel("Volume [ml]")
ax1.set_ylabel("Pressure [mmHg]")
ax1.grid(True)

for j in range(N_BEATS):
    v = ten[f'v_{j}'].detach().numpy()
    p_data = ten[f'p_{j}'].detach().numpy()
    ax1.plot(v, p_data, label=f'Data Fidelity Beat {j}', linestyle='--')
    
ax1.plot(ten[f'v_{0}'].detach().numpy(), u_pred.detach().numpy(), label='Real PV Data', color='black', linewidth=2)

ax1.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# Plot 3: Final Parameters PV Loops
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
    pp = torch.where(
        v <= v0_final,
        sn_final * torch.log((v - vd_final) / (v0_final - vd_final)),
        -sp_final * torch.log((vm_final - v) / (vm_final - v0_final))
    )
    pa = (ten['p_1'][0] - pp) * torch.exp(-t / tau_final)
    return pp + pa

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.set_title("Model PV Loops with Final Parameters - Case")
ax2.set_xlabel("Volume [ml]")
ax2.set_ylabel("Pressure [mmHg]")
ax2.grid(True)

for j in range(N_BEATS):
    v = ten[f'v_{j}']
    t = ten[f't_{j}']
    p = ten[f'p_{j}']
    p_model = pv_model_final(v, t)
    ax2.plot(v.detach().numpy(), p_model.detach().numpy(), label=f'Model Beat {j}', linewidth=2)
    ax2.plot(v.detach().numpy(), p.detach().numpy(), label='Real PV Data', color='black', linewidth=2)
ax2.legend()
plt.tight_layout()
plt.show()