import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Kaden Wines; Cont Mech
# Lab 2; Problem 1

# ---------------------------
# Experimental data
# ---------------------------
norm_radius = np.array([
    1.0074627, 1.0820896, 1.089552239, 1.1641791, 1.1641791, 
    1.1940299, 1.171641791, 1.246268657, 1.305970149, 1.634328358, 
    2.895522388, 3.444398022, 4.513349132, 4.869666169, 5.416018959, 
    5.772335996, 6.009880687, 6.413706662, 6.651251353, 6.829409871, 
    7.304499254, 7.185726908, 7.601430118, 7.898360982, 8.314064191
])
norm_pres = np.array([
    0.324815938, 0.396997257, 0.577450556, 0.685722535, 0.757903854,
    0.830085174, 1.010538473, 1.118810452, 1.190991771, 1.263173091,
    0.974447813, 0.866175834, 0.793994514, 0.757903854, 0.793994514,
    0.938357153, 1.010538473, 0.974447813, 1.046629132, 1.118810452,
    1.154901112, 1.227082431, 1.263173091, 1.299263751, 1.37144507

])

# ---------------------------
# Ogden Potential Function (N-term)
# ---------------------------
def ogden_pressure(lmbda, *params):
    
    N = len(params) // 2
    mu = np.array(params[0::2])
    alpha = np.array(params[1::2])
    
    # One-term pressure formula (simplified form)
    # P_i = mu_i * (lambda^(alpha_i - 1) - lambda^(-0.5*alpha_i - 1))
    # Then sum over all terms
    total = np.zeros_like(lmbda, dtype=float)
    for i in range(N):
        total += mu[i] * (lmbda**(alpha[i] - 1) - lmbda**(-0.5 * alpha[i] - 1))
    return total

# ---------------------------
# Number of terms to fit
# ---------------------------
N_terms = 3  # Change to 1, 2, 3... depending on how many you want

# Initial guesses for each term [mu1, alpha1, mu2, alpha2, ...]
initial_guess = []
for i in range(N_terms):
    initial_guess += [1.0, 2.0 + i]  # example: (mu, alpha) pairs

# ---------------------------
# Curve fit
# ---------------------------
popt, pcov = curve_fit(ogden_pressure, norm_radius, norm_pres, p0=initial_guess, maxfev=100000)

# Extract fitted parameters
mu = popt[0::2]
alpha = popt[1::2]

# ---------------------------
# Compute fitted curve
# ---------------------------
lambda_fit = np.linspace(min(norm_radius), max(norm_radius), 300)
pressure_fit = ogden_pressure(lambda_fit, *popt)

# ---------------------------
# Plot results
# ---------------------------
plt.figure(figsize=(7,5))
plt.plot(norm_radius, norm_pres, 'bo', label='Experimental Data')
plt.plot(lambda_fit, pressure_fit, 'r-', lw=2, label=f'Ogden Fit (N={N_terms})')

# Annotate fitted parameters
param_text = "\n".join([f"μ{i+1}={mu[i]:.3f}, α{i+1}={alpha[i]:.3f}" for i in range(N_terms)])
plt.text(0.05, 0.85, param_text, transform=plt.gca().transAxes, fontsize=10,
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

plt.title(f'N-term Ogden Model Fit (N={N_terms})')
plt.xlabel('Normalized Radius (λ)')
plt.ylabel('Gauge Pressure (psi)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Print fitted parameters
for i in range(N_terms):
    print(f"Term {i+1}: μ{i+1} = {mu[i]:.6f}, α{i+1} = {alpha[i]:.6f}")
