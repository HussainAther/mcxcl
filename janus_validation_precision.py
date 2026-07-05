import pmcxcl
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
SAVE_FILENAME = "xray_steering_precision.png"

# 1. Define High-Resolution 11-layer Grid
# Grid size expanded to 120x120x150 voxels to prevent boundary leakage
vol = np.ones((120, 120, 150), dtype='uint8')
for i in range(11):
    vol[:, :, (i*10):(i*10)+10] = i + 1

# 2. Set Precision Optical Properties for X-Ray Interaction
# [mua, mus, g, n]
# For X-rays, absorption is low, scattering is minimal/forward-directed, 
# and n is slightly less than 1.0 (1 - delta)
prop = [[0, 0, 1, 1.0]] # Layer 0: Vacuum/Air

# Small delta increments to test sub-millimeter refractive bending sensitivity
base_delta = 1e-5 
for i in range(11):
    n_xray = 1.0 - (i * base_delta)
    # Low absorption (mua=0.001), low scatter (mus=0.01), highly forward-directed (g=0.99)
    prop.append([0.001, 0.01, 0.99, n_xray])

# 3. Configure GPU Simulation Parameters
cfg = {
    'nphoton': 1e7,
    'vol': vol,
    'prop': prop,
    'srcpos': [60, 60, 1],   # Start at center of the grid
    'srcdir': [0.05, 0, 1],  # Micro-tilt vector to initiate interface angles
    'tstart': 0,
    'tend': 5e-9,
    'tstep': 5e-9,
    'gpuid': 1,              # Target Apple M1 GPU
    'isgpu': 1
}

# 4. Execute OpenCL Simulation Core
print("Launching X-Ray Precision Tracking on M1 GPU...")
res = pmcxcl.run(cfg)
fluence = res['flux'][:, :, :, 0]

# 5. Calculate Exit Centroid at Z=110 (Layer 11 Exit)
exit_slice = fluence[:, :, 110]
y_coords, x_coords = np.indices(exit_slice.shape)
total_fluence = np.sum(exit_slice)

if total_fluence > 0:
    centroid_x = np.sum(x_coords * exit_slice) / total_fluence
    shift_mm = centroid_x - 60
    print("\n--- SIMULATION RESULTS ---")
    print(f"Measured X-Ray Centroid Shift: {shift_mm:.6f} mm")
    print(f"Target Baseline: 11.577000 mm")
else:
    print("Error: No photon energy reached the exit layer.")

# 6. Save Spatial Fluence Profile
plt.figure(figsize=(10, 6))
plt.imshow(np.log10(fluence[60, :, :].T + 1e-10), aspect='auto', cmap='bone')
plt.colorbar(label='Log10 Photon Intensity')
if total_fluence > 0:
    plt.axvline(x=centroid_x, color='red', linestyle='--', label=f'Centroid: {centroid_x:.2f}mm')
plt.title('High-Precision X-Ray Path Tracking (Z-X Projection)')
plt.xlabel('X-axis (Lateral Profile)')
plt.ylabel('Z-axis (Depth Profile)')
plt.legend()
plt.savefig(SAVE_FILENAME)
print(f"Spatial profile successfully saved to: {SAVE_FILENAME}")
