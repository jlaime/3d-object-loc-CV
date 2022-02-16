import numpy as np

# Obj file
model_path = "Models/"
obj_file_name = "model_1_tri"
ext = ".obj"

# Capture file
data_path = "Data/"
capture_ixt = "capt_"
capture_ext = ".png"

# Camera view ranges
range_rho = np.linspace(7, 15, num = 5) #m
range_theta = range_phi = np.linspace(-50, 50, num = 51) #°

# Grad files
grad_ixt = "grad_"
grad_ext = ".pckl"

# Renders
render_path = "Others/"