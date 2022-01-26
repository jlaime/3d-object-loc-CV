"""

Pipeline :



> Camera calibration
	FOV
	Radial distortion
		Not needed here, since all the input testing data will be 3D generated, so perfect camera

> 3D model generation
	Object in center of the camera

	> Hierarchical view generation
		Define max view boundaries (d, omega, phi)
		Generate equally distributed camera positions

		For each pyramid level :
			Oversample views around each point
			For each group of cameras, compute similarity metric
			Pairs with highest similarity are merged
			Repeat until similarity < threshold
			
			Save the remaining views in pyramid
			For level 2+ : Save the child views references (Tree structure)
			-> used for refining

			Then: repeat on lower resolution render
				-> lower resolution = higher inplace similarity

			
	> Model image generation
		For each level, view in Tree :
			Orientate camera on object
			[ Similarity measure (c) robust to occlusion, clutter and non-linear contrast change
			Extract normals n = (x, y, z)
			Render each face with color = (x, y, z) normalized

			For normals 1, 2 :
				Compute color tensor C
				Edge amplitude A = eig(C) = sqrt(SUM(i1 - i2)², i in [x, y, z]
				Angle between 2 normals delta = 2arcsin(A/2)

			Filter edges with angle threshold

			Save all infos (pose, angle) in tree

> 3D object recognition
	Start at hisghest pyramid level
	Try to find the 2D matching model using similarity meeasure c
		Rot scale in range steps
		2D pose matches -> stored as candidates

	N-1 level
		Poses without parent are searched, and children of previous

	Apply homography (disto from camera center)
		Possibly precomputed

> Pose refinement
	Will see if done


> Evaluation

640 x 480p images, grayscale ?
theta = phi = [-50, +50]°
dist = [15, 30]cm 
f = 8.5mm
5 or 6 pyramid levels

				a --
             a aa a --
          a aa aaa aa a --
     a aa aaa aaaa aaa aa a --
a aa aaa aaaa aaaaa aaaa aaa aa a --
     a aa aaa aaaa aaa aa a --
          a aa aaa aa a --
             a aa a --
             	a --

"""

from offline_params import *
import cv2 as cv
from PIL import Image
import numpy as np


def grad_color(cm, cr, cc):
	"""Compute color edge amplitude and angle. cm = color_main, cr = color_row, cc = color_col (prev pixels)"""
	#		cc
	# cr	cm

	# BGR to RGB
	#cm = cm[::-1]
	#cr = cr[::-1]
	#cc = cc[::-1]

	cm = cm/255.0
	cr = cr/255.0
	cc = cc/255.0

	#if cm == (0, 0, 0) and cr == (0, 0, 0) and cc == (0, 0, 0):
	#	return (0, 0)

	# Row grad
	gr_r = cm[0] - cr[0]
	gr_g = cm[1] - cr[1]
	gr_b = cm[2] - cr[2]

	# Col grad
	gc_r = cm[0] - cc[0]
	gc_g = cm[1] - cc[1]
	gc_b = cm[2] - cc[2]

	if (gr_r, gr_g, gr_b, gc_r, gc_g, gc_b) == (0, 0, 0, 0, 0, 0):
		return 0.0

	# Intermediate grad
	grr = gr_r**2 + gr_g**2 + gr_b**2
	grc = gr_r*gc_r + gr_g*gc_g + gr_b*gc_b
	gcc = gc_r**2 + gc_g**2 + gc_b**2

	# Color tensor
	C = [[grr, grc], [grc, gcc]]

	eigen = np.linalg.eig(C)[0]
	#print(eigen)
	best_eig = max(eigen)

	A = np.sqrt(best_eig)

	return A


def similarity(m, s):
	n = len(grad_m)
	sim = 0
	for i in range(n):
		sim += np.abs(np.dot(m[i], s[i])) / (np.linalg.norm(m[i]) * np.linalg.norm(s[i]))

	return sim/n




capture_name = data_path + capture_ixt+obj_file_name+"_"+str(7.0)+"_"+str(-50.0)+"_"+str(-20.0)+".png"

img = cv.imread(capture_name)

cv.imshow("Display", img)
#cv.waitKey(0)

print(img)


edge = np.zeros(img.shape[:2], dtype=np.uint8)
	
#gray = cv.cvtColor(edge, cv.COLOR_BGR2GRAY)
print(edge.shape)


angle_threshold = np.radians(5.0)

for y in range(img.shape[0]-1):
	for x in range(img.shape[1]-1):
		amplitude = grad_color(img[y, x], img[y, x-1], img[y-1, x])

		if amplitude > 0.0:
			angle = 2*np.arcsin(amplitude/2)
			#print(angle)
			grayscale = 255 if angle > angle_threshold else 0
			edge.itemset(y, x, int(grayscale))

		#amplitude = 255 if angle > angle_threshold else 0
		


print("Lim", angle_threshold)

cv.imshow("Edge", edge)
cv.waitKey(0)


#kernel = np.ones((3, 3), np.uint8)
#edge_eroded = cv.erode(edge, kernel)
#cv.imshow("Eroded", edge_eroded)
#cv.waitKey(0)



