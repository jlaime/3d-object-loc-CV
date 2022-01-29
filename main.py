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
import imutils
import numpy as np
from skimage.measure import compare_ssim


def colornorm2amp(cm, cr, cc):
	"""Compute color edge amplitude. cm = color_main, cr = color_row, cc = color_col (prev pixels)"""
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


def image2edge(img, angle_threshold): # using colornorm2amp

	edge = np.zeros(img.shape[:2]) # , dtype=np.uint8
		
	#gray = cv.cvtColor(edge, cv.COLOR_BGR2GRAY)
	print(edge.shape)


	for y in range(1, img.shape[0]):
		for x in range(1, img.shape[1]):
			amplitude = colornorm2amp(img[y, x], img[y, x-1], img[y-1, x])

			if amplitude > 0.0:
				angle = 2*np.arcsin(amplitude/2)
				#print(angle)
				grayscale = amplitude if angle > angle_threshold else 0
				edge.itemset(y, x, grayscale)

			#amplitude = 255 if angle > angle_threshold else 0

	return edge



def F(theta, g11, g12, g22):
	return 1/2 * ((g11 + g22) + np.cos(2*theta)*(g11 - g22) + 2*g12*np.sin(2*theta))


def gray2grad(img): # taking gray image

	img = img / 255
	print(img)

	grad = np.zeros((img.shape[0], img.shape[1], 2))

	for y in range(img.shape[0]-1):
		for x in range(img.shape[1]-1):

			if img[y, x] == 0 and img[y+1, x] == 0 and img[y, x+1] == 0 and img[y+1, x+1] == 0:
				pass

			else :

				aj1 = (img[y, x+1] + img[y+1, x+1])/2 - (img[y, x] + img[y+1, x])/2
				aj2 = (img[y+1, x] + img[y+1, x+1])/2 - (img[y, x] + img[y, x+1])/2

				g11 = aj1**2
				g22 = aj2**2
				g12 = aj1*aj2


				if (g11 - g22) == 0.0:
					theta1 = np.pi/2 if g12 >= 0 else -np.pi/2
				else :
					theta1 = 1/2 * np.arctan(2*g12 / (g11 - g22))

				theta2 = theta1 + np.pi/2

				F1 = F(theta1, g11, g12, g22)
				F2 = F(theta2, g11, g12, g22)

				F_max = max(F1, F2)

				if F_max == F1:
					theta = theta1
				else :
					theta = theta2

				grad[y, x, 0] = np.sqrt(F_max*3)
				grad[y, x, 1] = theta

				#print(theta1, theta2)
				#grad[y, x, 1] = theta

	return grad


def color2grad(img): # taking 3 channel color image
		
	img = img / 255
	print(img)

	grad = np.zeros((img.shape[0], img.shape[1], 2))

	m = img.shape[2]

	for y in range(img.shape[0]-1):
		for x in range(img.shape[1]-1):

			g11 = 0
			g12 = 0
			g22 = 0

			for j in range(m):

				if img[y, x, j] == 0.0 and img[y+1, x, j] == 0.0 and img[y, x+1, j] == 0.0 and img[y+1, x+1, j] == 0.0:
					pass

				else :

					aj1 = (img[y, x+1, j] + img[y+1, x+1, j])/2 - (img[y, x, j] + img[y+1, x, j])/2
					aj2 = (img[y+1, x, j] + img[y+1, x+1, j])/2 - (img[y, x, j] + img[y, x+1, j])/2

					g11 += aj1**2
					g22 += aj2**2
					g12 += aj1*aj2

				#g11 = aj1**2
				#g22 = aj2**2
				#g12 = aj1*aj2

			#print(g11)


			if (g11 - g22) == 0.0:
				theta1 = np.pi/2 if g12 >= 0 else -np.pi/2
			else :
				theta1 = 1/2 * np.arctan(2*g12 / (g11 - g22))

			theta2 = theta1 + np.pi/2

			F1 = F(theta1, g11, g12, g22)
			F2 = F(theta2, g11, g12, g22)

			F_max = max(F1, F2)

			if F_max == F1:
				theta = theta1
			else :
				theta = theta2

			grad[y, x, 0] = np.sqrt(F_max)
			grad[y, x, 1] = theta

			#print(theta1, theta2)
			#grad[y, x, 1] = theta

	return grad





def similarity(m, s):
	n = len(grad_m)
	sim = 0
	for i in range(n):
		sim += np.abs(np.dot(m[i], s[i])) / (np.linalg.norm(m[i]) * np.linalg.norm(s[i]))

	return sim/n






def cosine_similarity(image, image2):
	flatten_image = image.flatten()
	flatten_image_2 = image2.flatten()

	flatten_image_norm = np.linalg.norm(flatten_image)
	flatten_image_2_norm = np.linalg.norm(flatten_image_2)

	dot_product = np.dot(flatten_image, flatten_image_2)

	print(dot_product, flatten_image_norm, flatten_image_2_norm)

	cosine_similarity = dot_product / (flatten_image_norm*flatten_image_2_norm)
	return cosine_similarity


def rotateImage(image, angle):
    row,col = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

"""
def similarity(img1, img2):
	flat1 = img1.flatten()
	flat2 = img2.flatten()

	n = flat1.shape[0]

	flatten_image_norm = np.linalg.norm(flatten_image)
	flatten_image_2_norm = np.linalg.norm(flatten_image_2)

	dot_product = np.dot(flatten_image, flatten_image_2)

	print(dot_product, flatten_image_norm, flatten_image_2_norm)

	cosine_similarity = dot_product / (flatten_image_norm*flatten_image_2_norm)
	return cosine_similarity
"""


angle_threshold = np.radians(5.0)
print("Lim", angle_threshold)


capture_name1 = data_path + capture_ixt+obj_file_name+"_"+str(7.0)+"_"+str(-50.0)+"_"+str(-20.0)+".png"
capture_name2 = data_path + capture_ixt+obj_file_name+"_"+str(7.0)+"_"+str(-50.0)+"_"+str(-24.0)+".png"

capture_name1 = data_path + "test2.png"
capture_name2 = data_path + "test1.png"


img1 = cv.imread(capture_name1)
img2 = cv.imread(capture_name2)
grad = gray2grad(cv.cvtColor(img1, cv.COLOR_BGR2GRAY))[:, :, 0]
color_grad = color2grad(img1)[:, :, 0]
cv.imshow("Display 1", img1)
cv.imshow("Display 2", img2)
cv.imshow("gray2grad", grad)
cv.imshow("color2grad", color_grad)
print(np.max(grad))
cv.waitKey(0)

edge1 = image2edge(img1, angle_threshold)
edge2 = image2edge(img2, 0)

cv.imshow("Color edge algorithm", edge2)
cv.waitKey(0)



for a in range(-10, 10):

	rotated = imutils.rotate(edge2, a)

	cv.imshow("Edge 1", rotated)
	#cv.imshow("Edge 2", rotated)
	cv.imshow("Difference", np.abs(rotated - color_grad))

	#print(cosine_similarity(edge1, rotated))
	#print(compare_ssim(edge1, rotated))


	cv.waitKey(0)


cv.waitKey(0)




#kernel = np.ones((3, 3), np.uint8)
#edge_eroded = cv.erode(edge, kernel)
#cv.imshow("Eroded", edge_eroded)
#cv.waitKey(0)



