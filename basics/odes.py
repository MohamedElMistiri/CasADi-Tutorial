'''
casADi's framework can be used to formulate linear and nonlinear systems, and solve the ordinary differential equations (ODEs) that arise from these systems. This module provides tools for defining ODEs, integrating them, and analyzing their behavior.

In this file, we will go through the formulation of different systems (e.g., Continuously Stirred Tank Reactor (CSTR),) and how to solve the ODEs that arise from these systems using CasADi's integrators. We will also show how to analyze the behavior of these systems by plotting their trajectories and phase portraits.

'''
# %% import dependencies
from casadi import *
import numpy as np
import matplotlib.pyplot as plt

# %% define the ODE for a CSTR system
# state variables
C_A = SX.sym('C_A') # concentration of A
T = SX.sym('T') # temperature

# control variable
T_j = SX.sym('T_j') # jacket temperature

# disturbance variable
C_Af = SX.sym('C_Af') # feed concentration of A
T_f = SX.sym('T_f') # feed temperature

# parameters 
# Case 1
F = 1 # flow rate (m^3/hr)
V = 1 # volume of the reactor (m^3)
ko = 14825*3600 # Arrhenius constant (1/h)
Delta_E = 11843 # activation energy (kcal/kg-mol)
Delta_H = -5215 # heat of reaction (kcal/kg-mol)
rhoCp = 500 # heat capacity (kcal/m^3-K)
UA = 250 # overall heat transfer coefficient (kcal/hr-K)
R = 1.98589 # gas constant (kcal/kg-mol-K)
# steady-state values
C_A_ss = 7.5709 # steady-state concentration of A (kg-mol/m^3)
T_ss = 315.0402 # steady-state temperature (K)

# Case 2
F = 1 # m^3/hr
V = 1 # m^3
ko = 9703*3600 # 1/hr
Delta_H = -5960 # kcal/kgmol
Delta_E = 11843 # kcal/kgmol
RhoCp = 500 # kcal/(m^3 C)
UA = 150 # kcal/(m^3 C hr)
R = 1.98589 # kcal/(kgmol K)
# Steady State Values:
C_A_ss = 8.5695 # kg-mol/m^3
T_ss  = 311.267 # K

# Initial conditions
C_Af0 = 10 # initial feed concentration of A (kg-mol/m^3)
T_f0 = 25 + 273.15 # initial feed temperature (K)
T_j0 = 25 + 273.15 # initial jacket temperature (K)
C_A0 = C_A_ss # initial concentration of A in the reactor (kg-mol/m^3)
T0 = T_ss # initial temperature in the reactor (K)

# define the ODEs
r_A = ko*exp(-Delta_E/(R*T))*C_A # reaction rate (kg-mol/m^3-hr)
dC_A_dt = F/V*(C_Af - C_A) - r_A # ODE for concentration of A
dT_dt = F/V*(T_f - T) + (-Delta_H/rhoCp)*r_A + UA/(V*rhoCp)*(T_j - T) # ODE for temperature

# define the state vector and the control/disturbance vector
x = vertcat(C_A, T) # state vector
u = vertcat(T_j) # control vector
d = vertcat(C_Af, T_f) # disturbance vector

# define the ODE function
f = vertcat(dC_A_dt, dT_dt) # ODE function
f_func = Function('f_func', [x, u, d], [f], ['x', 'u', 'd'], ['dx/dt']) # define a casadi function for the ODE
print(f_func)

# %% Linearize the ODE around the steady-state operating point
A = jacobian(f, x) # Jacobian of f with respect to x
B = jacobian(f, u) # Jacobian of f with respect to u
D = jacobian(f, d) # Jacobian of f with respect to d
A_func = Function('A_func', [x, u, d], [A], ['x', 'u', 'd'], ['A']) # define a casadi function for the A matrix
B_func = Function('B_func', [x, u, d], [B], ['x', 'u', 'd'], ['B']) # define a casadi function for the B matrix
D_func = Function('D_func', [x, u, d], [D], ['x', 'u', 'd'], ['D']) # define a casadi function for the D matrix
print(A_func)
print(B_func)
print(D_func)

# evaluate the A, B, and D matrices at the steady-state operating point
A_ss = A_func(vertcat(C_A_ss, T_ss), vertcat(T_j0), vertcat(C_Af0, T_f0)) # evaluate A at the steady-state operating point
B_ss = B_func(vertcat(C_A_ss, T_ss), vertcat(T_j0), vertcat(C_Af0, T_f0)) # evaluate B at the steady-state operating point
D_ss = D_func(vertcat(C_A_ss, T_ss), vertcat(T_j0), vertcat(C_Af0, T_f0)) # evaluate D at the steady-state operating point
print('A matrix at steady-state operating point:\n', A_ss)
print('B matrix at steady-state operating point:\n', B_ss)  
print('D matrix at steady-state operating point:\n', D_ss)

# linearized ODE function
f_lin = A_ss @ (x - vertcat(C_A_ss, T_ss)) + B_ss @ (u - vertcat(T_j0)) + D_ss @ (d - vertcat(C_Af0, T_f0)) # linearized ODE function
f_lin_func = Function('f_lin_func', [x, u, d], [f_lin], ['x', 'u', 'd'], ['dx/dt']) # define a casadi function for the linearized ODE
print(f_lin_func)

# %% define the integrator for the ODE
dt = 0.01 # time step for integration (hr)
int_options = {
    'abstol': 1e-8,  # Absolute tolerance
    'reltol': 1e-8,  # Relative tolerance
    'max_num_steps': 1000  # Maximum number of steps per integration
    } # options for the integrator

integ = integrator('integ', 'idas',
                {
                    'x': x, 
                    'p': vertcat(u, d),
                    'ode': f
                },
                0, dt, int_options
            ) # define an integrator for the ODE

print(integ)

integ_lin = integrator('integ_lin', 'idas',
                {
                    'x': x, 
                    'p': vertcat(u, d),
                    'ode': f_lin
                },
                0, dt, int_options
            ) # define an integrator for the linearized ODE

print(integ_lin)

# %% simulate the system for a given control and disturbance trajectory
# time horizon for simulation
T_sim = 30 # simulation time (hr)
N_sim = int(T_sim/dt) # number of simulation steps

# define the control and disturbance trajectories
T_j_traj = np.ones(N_sim)*T_j0 # jacket temperature trajectory (K)
C_Af_traj = np.ones(N_sim)*C_Af0 # feed concentration trajectory (kg-mol/m^3)
T_f_traj = np.ones(N_sim)*T_f0 # feed temperature trajectory (K)

# T_j_traj[int(1/dt):] = T_j0 + 9
# T_f_traj[int(1/dt):] = T_f0 + 4
C_Af_traj[int(1/dt):] = C_Af0 + 2

# initialize arrays to store the simulation results
C_A_traj = np.zeros(N_sim+1) # array to store concentration of A trajectory
T_traj = np.zeros(N_sim+1) # array to store temperature trajectory
C_A_traj[0] = C_A0 # set initial concentration of A
T_traj[0] = T0 # set initial temperature

C_A_traj_lin = np.zeros(N_sim+1) # array to store concentration of A trajectory for linearized system
T_traj_lin = np.zeros(N_sim+1) # array to store temperature trajectory for linearized system
C_A_traj_lin[0] = C_A0 # set initial concentration of A for linearized system
T_traj_lin[0] = T0 # set initial temperature for linearized system

# simulate the system
for i in range(N_sim):
    # get the current state
    x_current = vertcat(C_A_traj[i], T_traj[i])
    x_current_lin = vertcat(C_A_traj_lin[i], T_traj_lin[i])
    
    # get the current control and disturbance
    u_current = vertcat(T_j_traj[i])
    d_current = vertcat(C_Af_traj[i], T_f_traj[i])
    
    # integrate the ODE to get the next state
    res = integ(x0=x_current, p=vertcat(u_current, d_current))
    x_next = res['xf'] # next state after integration

    # integrate the linearized ODE to get the next state for the linearized system
    res_lin = integ_lin(x0=x_current_lin, p=vertcat(u_current, d_current))
    x_next_lin = res_lin['xf'] # next state after integration for the linearized system

    # store the next state in the trajectory arrays
    C_A_traj[i+1] = x_next[0]
    T_traj[i+1] = x_next[1]
    C_A_traj_lin[i+1] = x_next_lin[0]
    T_traj_lin[i+1] = x_next_lin[1]

# % plot the results
time = np.linspace(0, T_sim, N_sim+1) # time array for plotting

# plot state trajectories
plt.figure(figsize=(7, 5))
plt.subplot(2, 1, 1)
plt.plot(time, C_A_traj, linewidth=1.5, label='Nonlinear System', color='black')
plt.plot(time, C_A_traj_lin, linewidth=1.5, linestyle='--', color='blue', label='Linearized System')
plt.xlabel('Time (hr)')
plt.ylabel('$C_A$ ($kg mol/m^3$)')
plt.title('Simulation Results')
plt.xlim(0, T_sim)
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, T_traj, linewidth=1.5, label='Nonlinear System', color='black')
plt.plot(time, T_traj_lin, linewidth=1.5, linestyle='--', color='blue', label='Linearized System')
plt.xlabel('Time (hr)')
plt.ylabel('$T$ ($K$)')
# plt.title('Simulation Results')
plt.xlim(0, T_sim)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# plot control and disturbance trajectories
plt.figure(figsize=(7, 5))
plt.subplot(2, 1, 2)
plt.plot(time[:-1], T_j_traj, label='$T_j$ (K)', linewidth=1.5)
plt.plot(time[:-1], T_f_traj, label='$T_f$ (K)', linewidth=1.5, linestyle='--')
plt.xlabel('Time (hr)')
plt.ylabel('$T$ ($K$)')

plt.xlim(0, T_sim)
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 1)
plt.plot(time[:-1], C_Af_traj, label='$C_{Af}$ (kg-mol/m^3)', linewidth=1.5)
plt.xlabel('Time (hr)')
plt.ylabel('$C_{Af}$ ($kg-mol/m^3$)')
plt.title('Control and Disturbance Trajectory')
# plt.title('Disturbance Trajectory')
plt.xlim(0, T_sim)
plt.grid(True)
plt.tight_layout()
plt.show()

# plot phase portrait
plt.figure(figsize=(8, 6))
plt.plot(C_A_traj, T_traj, linewidth=1.5)
plt.xlabel('$C_A$ ($kg-mol/m^3$)')
plt.ylabel('$T$ ($K$)')
plt.title('Phase Portrait')
plt.grid(True)
plt.show()

# %%
