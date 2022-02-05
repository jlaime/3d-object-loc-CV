"""
import 3D obj file
render obj at multiple angles with colored faces
save pictures


"""
"""
pip install PIL 
pip install vpython
"""

from PIL import Image, ImageGrab
from vpython import *
from wavefront import *
from offline_params import *
import numpy as np
import math
import time, sys


# Capture
res = (640, 480) #px
capture_origin = (54, 180)

# Camera
cam_fov = 60 #°
cam_loc = [0, 0, 0]		# x y z

# init
cam_rho = 0				# dist
cam_theta = 0			# angle plat
cam_phi = 0				# angle vertical

# Object
scale = 0.01 # meter -> cm
act_rot = 0
faces = []
normals = []

vertex_list = []
tri_list = []

def sph2cart(r, theta, phi):
    return [
         r * math.sin(theta) * math.cos(phi),
         r * math.sin(theta) * math.sin(phi),
         r * math.cos(theta)
    ]


def normal2color(n):
	return vector((1+n[0])/2., (1+n[1])/2., (1+n[2])/2.)


def rotate(current, new):
	return new - current


# Import obj

model = load_obj(model_path + obj_file_name + ext, triangulate=True)


#print(model.vertices)
# List of vect pos: [x, y, z]

#print(model.normals)
# List of normals vectors: [x, y, z]

#print(model.polygons)
# List of faces: [(vect_id, -1, normal_id)*nb of vectices in face]


for fi in model.polygons: 
	# fi=[(vect_id, -1, normal_id) * nb of vectices in face]
	# v=(vect_id, -1, normal_id)

	# f=[(x_i, y_i, z_i) * nb of vectices in face]
	f = [model.vertices[v[0]] for v in fi]
	faces.append(f)

	# Face normal = avg of vertex normals vectors (model.normals[v[2]])
	# n = (x_avg, y_avg, z_avg)
	n = np.sum([model.normals[v[2]] for v in fi], axis=0) / float(len(fi))
	normals.append(n)


print("First face", faces[0], "Associated normal", normals[0])
print("Number of faces/normals", len(faces), len(normals))


# Displaying scene

scene2 = canvas(title='2D model data generation',
    width=res[0]/1.25, height=res[1]/1.25,
    center=vector(0,0,0), background=color.black, ambient=color.white)

scene2.exit = 1

scene2.lights = []

scene2.fov = radians(cam_fov)


# Object generation

for i in range(len(faces)):
	f = faces[i]
	color_face = normal2color(normals[i])
	vertex_list.append([vertex( pos=vector(vex[0], vex[1], vex[2]), color=color_face ) for vex in f])

for i in range(len(vertex_list)):
	# Assuming all faces are triangles
	tri_list.append( triangle(vs=vertex_list[i]) )

obj = compound(tri_list)

obj.pos = vector(0, 0, 0)
print(obj.pos, obj.axis)


# Repositionment & config

time.sleep(1)

obj.rotate(radians(90), vector(0, 1, 0))
#obj.rotate(radians(90), vector(0, 1, 0))
#obj.rotate(radians(90), vector(0, 0, 1))

scene2.autoscale = 0
scene2.center = vector(0, 0, 0)

scene2.up = vector(0, 0, 1)


# Buffer

for x in range(50):
	print(x)
	rate(5)


# Training data generation

for rho in range_rho:
	for theta in range_theta:
		for phi in range_phi:

			coo = sph2cart(rho, radians(95-theta), radians(phi))
			scene2.camera.pos = vector(coo[0], coo[1], coo[2])
			scene2.camera.axis = vector(-coo[0], -coo[1], -coo[2])

			time.sleep(0.05)
			#rate(10)

			# Capture

			capture_name = capture_ixt+obj_file_name+"_"+str(rho)+"_"+str(theta)+"_"+str(phi)+".png"
			capture = ImageGrab.grab((capture_origin[0], capture_origin[1], capture_origin[0]+res[0], capture_origin[1]+res[1]))

			capture.save(data_path+capture_name, 'PNG')

			print(rho, theta, phi)

			#scene2.capture("visual_python_capture")

#scene2.capture("visual_python_capture")

sys.exit()


#scene2.forward = vector(-1, 1, 1)
#scene2.range = 10