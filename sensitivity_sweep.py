import pmcxcl
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
TUMOR_SIZES = [2.0, 1.5, 1.0, 0.5] # mm
RESULTS = {}

def run_simulation(size):
    vol = np.ones((100, 100, 120), dtype='uint8')
    for i in range(11):
        vol[:, :, (i*10):(i*10)+10] = i + 1
    
    # Add tumor of variable size 's' at Z=60
    s = int(size / 2)
    if s < 1: s = 1 # Minimum 1 voxel
    vol[50-s:50+s, 50-s:50+s, 59:61] = 12 

    prop = [[0, 0, 1, 1.0]] # Air
    for i in range(11):
        prop.append([0.01, 1.0, 0.9, 1.33 + (i * 0.08)]) # Steering
    prop.append([0.5, 1.0, 0.9, 1.40]) # Tumor

    cfg = {'nphoton': 1e7, 'vol': vol, 'prop': prop, 
           'srcpos': [50, 50, 1], 'srcdir': [0.1, 0, 1], 
           'gpuid': 1, 'isgpu': 1}
    
    return pmcxcl.run(cfg)['flux'][:, :, :, 0]

# Baseline (No Tumor)
flux_h = run_simulation(0)

# Loop through sizes
for size in TUMOR_SIZES:
    print(f"Testing {size}mm tumor...")
    flux_t = run_simulation(size)
    # Calculate peak contrast signal
    contrast = np.max((flux_h - flux_t) / (flux_h + 1e-10))
    RESULTS[size] = contrast

# Plot the Sensitivity Curve
plt.figure(figsize=(8, 5))
plt.plot(list(RESULTS.keys()), list(RESULTS.values()), 'o-', linewidth=2)
plt.title("Janus Sphere: Detection Sensitivity Limit")
plt.xlabel("Tumor Diameter (mm)")
plt.ylabel("Signal Contrast (Shadow Strength)")
plt.grid(True)
plt.savefig("sensitivity_limit.png")
print("Data table generated and figure saved.")
