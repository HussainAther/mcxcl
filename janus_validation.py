import pmcxcl
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
SAVE_FILENAME = "janus_lateral_shift.png"
ADD_TUMOR = False  # Set to True to see the 2mm shadow signal

# 1. Define the 11-layer Janus Geometry (10mm per layer)
# Domain size 60x60x120 voxels (1 voxel = 1mm)
vol = np.ones((60, 60, 120), dtype='uint8')
for i in range(11):
    vol[:, :, (i*10):(i*10)+10] = i + 1

# Add a 2mm tumor if toggled
if ADD_TUMOR:
    # Placing a 2mm cube at Z=55 (middle of steering path)
    vol[29:31, 29:31, 54:56] = 12 

# 2. Set Optical Properties for each layer
# [mua, mus, g, n]
# Realistic tissue-like properties to ensure data normalization works
# Increase mua to 0.01 and mus to 1.0
prop = [[0, 0, 1, 1.0]] # Layer 0: Air
for i in range(11):
    # n increases from 1.33 to 1.43 to steer the beam
    prop.append([0.01, 1.0, 0.9, 1.33 + (i * 0.01)])

# If tumor exists, make it highly absorbing (mua=0.5)
if ADD_TUMOR:
    prop.append([0.5, 1.0, 0.9, 1.35])

# 3. Configure Simulation
cfg = {
    'nphoton': 1e7,
    'vol': vol,
    'prop': prop,
    'srcpos': [30, 30, 1],
    'srcdir': [0, 0, 1],
    'tstart': 0,
    'tend': 5e-9,
    'tstep': 5e-9,
    'gpuid': 1,
    'isgpu': 1
}

# 4. Run Simulation
print(f"Launching Janus Simulation (Tumor={ADD_TUMOR})...")
res = pmcxcl.run(cfg)
fluence = res['flux'][:, :, :, 0]

# 5. Calculate Exit Centroid
exit_slice = fluence[:, :, 110]
y_coords, x_coords = np.indices(exit_slice.shape)
total_fluence = np.sum(exit_slice)
centroid_x = np.sum(x_coords * exit_slice) / total_fluence
shift_mm = centroid_x - 30

print(f"Simulation Complete!")
print(f"Measured Lateral Shift: {shift_mm:.4f} mm")

# 6. Save Visualization
plt.figure(figsize=(10, 6))
# Plotting the Z-X cross section
plt.imshow(np.log10(fluence[30, :, :].T + 1e-10), aspect='auto', cmap='hot')
plt.colorbar(label='Log10 Photon Flux')
plt.axvline(x=centroid_x, color='cyan', linestyle='--', label=f'Centroid: {centroid_x:.2f}mm')
plt.title(f'Janus Steering Path (Shift: {shift_mm:.3f} mm)')
plt.xlabel('X-axis (Lateral Shift)')
plt.ylabel('Z-axis (Depth)')
plt.legend()

# Save the file instead of showing it
plt.savefig(SAVE_FILENAME)
print(f"Figure saved as: {SAVE_FILENAME}")
