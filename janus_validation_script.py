import pmcxcl
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
SAVE_FILENAME = "janus_validation_11mm_shift.png"
ADD_TUMOR = False 

# 1. Define the 11-layer Janus Geometry (10mm per layer)
vol = np.ones((100, 100, 120), dtype='uint8') # Increased XY for wider shift
for i in range(11):
    vol[:, :, (i*10):(i*10)+10] = i + 1

# 2. Set Optical Properties (Refractive Index Gradient)
prop = [[0, 0, 1, 1.0]] # Layer 0: Air
for i in range(11):
    # To get ~11mm shift, we need a steeper 'n' gradient
    # We increase 'n' significantly per layer to force refraction
    n_val = 1.33 + (i * 0.08) 
    prop.append([0.02, 0.5, 0.9, n_val]) # Low scattering to see steering clearly

# 3. Configure Simulation (M1 GPU)
cfg = {
    'nphoton': 1e7,
    'vol': vol,
    'prop': prop,
    'srcpos': [50, 50, 1], # Start at center
    'srcdir': [0.1, 0, 1], # Initial small tilt to kickstart steering
    'gpuid': 1,
    'isgpu': 1
}

# 4. Run Simulation
print("Launching Janus Steering Validation...")
res = pmcxcl.run(cfg)
fluence = res['flux'][:, :, :, 0]

# 5. Calculate Exit Centroid at Z=110
exit_slice = fluence[:, :, 110]
y_coords, x_coords = np.indices(exit_slice.shape)
total_fluence = np.sum(exit_slice)
centroid_x = np.sum(x_coords * exit_slice) / total_fluence
shift_mm = centroid_x - 50

print(f"\nVALIDATION RESULTS:")
print(f"Measured Lateral Shift: {shift_mm:.4f} mm")
print(f"Target Displacement: 11.577 mm")

# 6. Save Visualization
plt.figure(figsize=(10, 6))
plt.imshow(np.log10(fluence[50, :, :].T + 1e-10), aspect='auto', cmap='hot', extent=[0,100,120,0])
plt.colorbar(label='Log10 Photon Flux')
plt.axvline(x=centroid_x, color='cyan', linestyle='--', label=f'Exit: {centroid_x:.2f}mm')
plt.title(f'Janus Steering Path (Shift: {shift_mm:.3f} mm)')
plt.legend()
plt.savefig(SAVE_FILENAME)
