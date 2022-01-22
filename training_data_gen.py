"""
import 3D obj file
render obj at multiple angles with colored faces
save pictures


"""

from vpython import *
from wavefront import *
from offline_params import *
import numpy as np
import math

# Obj file
path = "Models/"
obj_file_name = "model_1_tri"
ext = ".obj"

# Capture file
capture_path = "C:/Users/jlm/Downloads/"
capture_name = "visual_python_capture"
capture_ext = ".png"

# Capture
res = (640, 480) #px

# Camera
cam_fov = 60 #°
cam_loc = [0, 0, 0]		# x y z

# init
cam_rho = 0				# dist
cam_theta = 0			# angle plat
cam_phi = 0				# angle vertical

range_rho = np.linspace(15, 25, num = 50, endpoint=False) #m
range_theta = range_phi = np.linspace(0, 180, num = 50, endpoint=False) #°

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


# IMPORT OBJ

model = load_obj(path + obj_file_name + ext, triangulate=True)


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


# Displaying
scene2 = canvas(title='2D model data generation',
    width=640, height=480,
    center=vector(0,0,0), background=color.black, ambient=color.white)

scene2.lights = []



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

#obj.rotate(radians(-90))

scene2.autoscale = 0
scene2.center = vector(0, 0, 0)

scene2.up = vector(0, 0, 1)

#for rho in range_rho:
for theta in range_theta:
	print("theta", 90-theta/2)
	for phi in range_phi:
	
		

		coo = sph2cart(15, radians(90-theta/2), radians(phi*2))
		scene2.camera.pos = vector(coo[0], coo[1], coo[2])
		scene2.camera.axis = vector(-coo[0], -coo[1], -coo[2])
		

		rate(60)
			



scene2.capture("visual_python_capture")

#scene2.forward = vector(-1, 1, 1)
#scene2.range = 10


#scene.capture("visual_python_capture")