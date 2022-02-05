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
import time, sys

import cProfile

import matplotlib.pyplot as plt

from itertools import combinations
import pickle
from scipy.sparse import csr_matrix

from treelib import Tree, Node


def fast_eigen(a, b, c, d):
	"""
	:param a: first coefficient of the 2*2 matrix
	:param b:
	:param c:
	:param d:
	:return: eigen values of the 2*2 matrix
	"""
	# https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html

	T = a+d # trace
	D = a*d - b*c # determinant

	L1 = T/2 + np.sqrt(T**2/(4-D))
	L2 = T/2 - np.sqrt(T**2/(4-D))

	return L1, L2





def colornorm2amp(cm, cr, cc):
	"""Compute color edge amplitude. cm = color_main, cr = color_row, cc = color_col (prev pixels)
	:param cm: color_main of the pixel
	:param cr: color_row (previous pixel)
	:param cc: color_col (previous pixel)
	:return: the amplitude of the pixel as defined in the paper (on which we then filter to remove edges we don't care about
	-> if angle (proportional to amplitude) > threshold_angle -> amplitude replaced by 0)

		"""

	#		cc
	# cr	cm

	# BGR to RGB
	#cm = cm[::-1]
	#cr = cr[::-1]
	#cc = cc[::-1]

	#if cm == (0, 0, 0) and cr == (0, 0, 0) and cc == (0, 0, 0):
	#	return (0, 0)

	#if not(cm[0] - cr[0] == 0 and cm[1] - cr[1] == 0 and cm[2] - cr[2] == 0 and cm[0] - cc[0] == 0 and cm[1] - cc[1] == 0 and cm[2] - cc[2] == 0):

		#gr_r /= 255 ; gr_g /= 255 ; gr_b /= 255 ; gc_r /= 255 ; gc_g /= 255 ; gc_b /= 255

	# Row grad
	gr_r = cm[0] - cr[0]
	gr_g = cm[1] - cr[1]
	gr_b = cm[2] - cr[2]

	# Col grad
	gc_r = cm[0] - cc[0]
	gc_g = cm[1] - cc[1]
	gc_b = cm[2] - cc[2]

	# Intermediate grad
	grr = gr_r**2 + gr_g**2 + gr_b**2
	grc = gr_r*gc_r + gr_g*gc_g + gr_b*gc_b
	gcc = gc_r**2 + gc_g**2 + gc_b**2

	# Color tensor
	#C = [[grr, grc], [grc, gcc]]

	eigen = fast_eigen(grr, grc, grc, gcc)

	#eigen = np.linalg.eig(C)[0]
	#print(eigen)
	best_eig = max(eigen)

	A = np.sqrt(best_eig) #amplitude

	return A



def image2edge(img, angle_threshold): # using colornorm2amp
	"""

	:param img: image on which we want to filter edges
	:param angle_threshold: the threshold condition for filtering (the bigger, the more edges we keep)
	:return: matrix of shape (image_width, image_lenght, 2) with the amplitude and angle values for each pixel
	(used for edges filtering)
	"""
	img = img / 255

	edge = np.zeros(img.shape[:2], 2) # , dtype=np.uint8
		
	#gray = cv.cvtColor(edge, cv.COLOR_BGR2GRAY)
	print(edge.shape)

	for y in range(1, img.shape[0]):
		for x in range(1, img.shape[1]):

			if not((img[y, x, 0] == img[y, x-1, 0] and img[y, x, 1] == img[y, x-1, 1] and img[y, x, 2] == img[y, x-1, 2]) and 
				(img[y, x, 0] == img[y-1, x, 0] and img[y, x, 1] == img[y-1, x, 1] and img[y, x, 2] == img[y-1, x, 2])):

				amplitude = colornorm2amp(img[y, x], img[y, x-1], img[y-1, x])
				angle = 2*np.arcsin(amplitude/2)
				#print(angle)
				edge[y, x, 0] = amplitude if angle > angle_threshold else 0
				edge[y, x, 1] = angle

				#amplitude = 255 if angle > angle_threshold else 0

	return edge



def F(theta, g11, g12, g22):
	"""

	:param theta: angle (variable)
	:param g11:
	:param g12:
	:param g22:
	:return: function that we want to maximize, used in gray2grad() function
		"""
	return 1/2 * ((g11 + g22) + np.cos(2*theta)*(g11 - g22) + 2*g12*np.sin(2*theta))


def gray2grad(img): # taking gray image
	"""

	:param img: grayscale image
	:return: matrix of shape (image_width, image_lenght, 2) with the amplitude and angle values for each pixel (later used
	for cosine similarity)
	"""
	img = img / 255
	#print(img)

	grad = np.zeros((img.shape[0], img.shape[1], 2))

	for y in range(img.shape[0]-1):
		for x in range(img.shape[1]-1):

			if not(img[y, x] == img[y+1, x] and img[y, x] == img[y, x+1] and img[y, x] == img[y+1, x+1]):

				aj1 = (img[y, x+1] + img[y+1, x+1])/2 - (img[y, x] + img[y+1, x])/2
				aj2 = (img[y+1, x] + img[y+1, x+1])/2 - (img[y, x] + img[y, x+1])/2

				g11 = aj1**2
				g22 = aj2**2
				g12 = aj1*aj2

				if g11 - g12 != 0 and g11 - g22 != 0:


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
					#grad.itemset(y, x, 0, np.sqrt(F_max*3))
					#grad.itemset(y, x, 1, theta)

					#print(theta1, theta2)
					#grad[y, x, 1] = theta

	return grad


def color2grad(img): # taking 3 channel color image
	"""
	Idem gray2grad() function for a colored image.
	:param img: colored image
	:return: matrix of shape (image_width, image_lenght, 2) with the amplitude and angle values for each pixel (later used
	for cosine similarity)
	"""
	img = img / 255
	#print(img)

	grad = np.zeros((img.shape[0], img.shape[1], 2))

	m = img.shape[2]

	for y in range(img.shape[0]-1):
		for x in range(img.shape[1]-1):

			g11 = 0
			g12 = 0
			g22 = 0

			for j in range(m):

				if not(img[y, x, j] == img[y+1, x, j] and img[y, x, j] == img[y, x+1, j] and img[y, x, j] == img[y+1, x+1, j]):

					aj1 = (img[y, x+1, j] + img[y+1, x+1, j])/2 - (img[y, x, j] + img[y+1, x, j])/2
					aj2 = (img[y+1, x, j] + img[y+1, x+1, j])/2 - (img[y, x, j] + img[y, x+1, j])/2

					g11 += aj1**2
					g22 += aj2**2
					g12 += aj1*aj2

				#g11 = aj1**2
				#g22 = aj2**2
				#g12 = aj1*aj2

			#print(g11)

			if g11 - g12 != 0 and g11 - g22 != 0:

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
				#grad.itemset(y, x, 0, np.sqrt(F_max))
				#grad.itemset(y, x, 1, theta)

				#print(theta1, theta2)
				#grad[y, x, 1] = theta

	return grad



def grad2cart(grad):
	"""

	:param grad: angles and amplitudes matrix (like the output of color2grad() function for instance)
	:return: matrix of same shape (x,y) instead of (amplitude, angle)
	"""
	cart = np.zeros((grad.shape[0], grad.shape[1], 2))

	for y in range(grad.shape[0]):
		for x in range(grad.shape[1]):
			#cart.itemset(y, x, )
			cart[y, x] = vec2cart(grad[y, x])

	return cart



def vec2cart(vec):
	"""

	:param vec: vector defined by (amplitude, angle)
	:return: vector (x,y)
	"""
	return (vec[0] * np.cos(vec[1]), vec[0] * np.sin(vec[1]))


def norm(cart):
	"""

	:param cart: vector (x,y)
	:return: norm of the vector
	"""
	return np.sqrt( cart[0]**2 + cart[1]**2 )


def similarity(m, s):
	"""

	:param m: x and y matrix (or "gradients matrix") of 1st image
	:param s: x and y matrix (or "gradients matrix") of 2nd image
	:return: cosine similarity of the two images
	"""
	sim = 0
	n = 0
	for y in range(m.shape[0]):
		for x in range(m.shape[1]):

			if m[y, x, 0] != 0 and m[y, x, 1] != 0:
				if s[y, x, 0] != 0 and s[y, x, 1] != 0:
					sim += np.abs(np.dot(m[y, x], s[y, x])) / (norm(m[y, x]) * norm(s[y, x]))
					n+=1
			
	return sim/n


def similarity_vec(m, s, correction = 0):
	"""

	:param m: angles and amplitudes matrix (or "gradients matrix") of 1st image
	:param s: angles and amplitudes matrix (or "gradients matrix") of 2nd image
	:return: cosine similarity of the two images
	"""
	sim = 0
	n = 0
	for y in range(m.shape[0]):
		for x in range(m.shape[1]):

			if m[y, x, 0] != 0 and s[y, x, 0] != 0:
					sim += abs(np.cos(s[y, x, 1] - m[y, x, 1] + correction))
					n+=1
	

	return sim/n


def similarity_sparse(m, s): # nul a chier
	"""

	:param m: sparsed matrix of 1st image
	:param s: sparsed matrix of 2nd image
	:return: cosine similarity of the two images
	"""
	sim = 0
	n = 0

	for e in m[1:]:
		y, x, t = e

		for f in s[1:]:

			if y == f[0] and x == f[1]:
				sim += abs(np.cos(f[2][1] - t[1]))
				n+=1

				break

			if f[0] > y and f[1] > x:
				#print("Break")
				break

		#match = [f for f in s[1:] if (y == f[0] and x == f[1])]

		#print(match)

		#if len(match) == 1:
		#	match = match[0]

		#	sim += abs(np.cos(match[2][1] - t[1]))
		#	n+=1


	return sim/n


# def cosine_similarity(image, image2):
# 	flatten_image = image.flatten()
# 	flatten_image_2 = image2.flatten()

# 	flatten_image_norm = np.linalg.norm(flatten_image)
# 	flatten_image_2_norm = np.linalg.norm(flatten_image_2)

# 	dot_product = np.dot(flatten_image, flatten_image_2)

# 	print(dot_product, flatten_image_norm, flatten_image_2_norm)

# 	cosine_similarity = dot_product / (flatten_image_norm*flatten_image_2_norm)
# 	return cosine_similarity


def rotateImage(image, angle):
    """

    :param image: image we want to rotate
    :param angle: angle of rotation
    :return: rotated image
    """
    row,col = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image



def grad_render(img, grad, color, arrow_len, step):
	"""
	Draw gradient vectors (or arrows) directly onto the image
	:param img: image
	:param grad: amplitudes and angles matrix
	:param color:
	:param arrow_len: lenght of the gradient vectors
	:param step: how many pixels we move when moving across the image (so that we don't draw arrows for every pixel)
	"""
	for y in range(0, img.shape[0], step):
		for x in range(0, img.shape[1], step):

			vec_x, vec_y = vec2cart(grad[y, x])

			if grad[y, x, 0] > 0 :

				len_x = int(arrow_len * vec_x)
				len_y = int(arrow_len * vec_y)

				#print(len_x, len_y)
				
				cv.arrowedLine(img, (x, y), (x+len_x, y+len_y), color)



def grad2sparse(grad):
	"""

	:param grad: amplitudes and angles matrix
	:return: sparsed matrix
	"""
	#t1 = time.time()
	sparse_array = [(grad.shape[0], grad.shape[1], grad.shape[2])]

	for y in range(grad.shape[0]):
		for x in range(grad.shape[1]):

			if grad[y, x, 0] > 0:
				sparse_array.append((y, x, grad[y, x]))

	#print(time.time() - t1, len(sparse_array))
	return sparse_array


def sparse2grad(sparse):
	"""

	:param sparse: sparsed matrix
	:return: amplitudes and angles matrix
	"""
	#t1 = time.time()
	grad = np.zeros(sparse[0])

	for e in sparse[1:]:
		grad[e[0], e[1]] = e[2]

	#print(time.time() - t1)

	return grad


def save_sparse_grad():
	"""
	Function to save the gradient matrices so that we don't recalculate them everytime the code is run. Each gradient matrix
	is saved as one file. Since most gradients inside a gradient matrix are 0, we save a sparse version of the gradient matrix
	that is smaller in size.

	"""
	for rho in range_rho:
		for theta in range_theta:
			for phi in range_phi:

				t1 = time.time()

				store_name = data_path + grad_ixt+obj_file_name+"_"+str(rho)+"_"+str(theta)+"_"+str(phi)+"_"+str(5)+grad_ext


				try :
					f = open(store_name, 'rb')
					#grad_captures = pickle.load(f)
					f.close()
					print("Gradient deja existant !")

				except Exception as e:
					print(e)

					print("Generation gradient...")

					capture_name = data_path + capture_ixt+obj_file_name+"_"+str(rho)+"_"+str(theta)+"_"+str(phi)+".png"
					print(capture_name)
					img = cv.imread(capture_name, cv.IMREAD_GRAYSCALE)
					grad = gray2grad(img)

					sparse = grad2sparse(grad)
					#print(sparse2grad(sparse))

					f = open(store_name, 'wb')
					pickle.dump(sparse, f)
					f.close()



#save_sparse_grad()
#sys.exit()



"""

Create tree holder

Level -1:
All views

Level 0:

Subsample ranges
For each view, access all neighbours views
	For each pair of views, compute similarity
	If sim > thresh -> merge

	If no pair > thresh remaining -> Save all reamining views and main view to level, with ref to merged views

Level 1:

Downsample picture resolution
Repeat

"""


def pos2name(pos):
	"""

	:param pos: (pho, theta, phi)
	:return: string version of the pos
	"""
	return str(int(pos[0]))+","+str(int(pos[1]))+","+str(int(pos[2]))

def name2pos(name):
	return name.split(",")


def gen_lvl_0(step, sim_thresh, pas):

	rg = np.arange(0, int(step*pas), pas)
	print(rg)
	pairs = np.linspace((0, 0), (int(step*pas), int(step*pas)), int(pas))

	tree_list_0 = []

	# Subsampling
	for rho in range_rho:
		for theta in range_theta[0:-1:step]:
			for phi in range_phi[0:-1:step]:

				t1 = time.time()

				# List of saved gradients [(r, t, p), grad] 
				grad_captures = []

				# Importing gradients files for each position in the range
				for i in range(step):
					for j in range(step):

						t = theta + rg[i]
						p = phi + rg[j]

						store_name = data_path + grad_ixt+obj_file_name+"_"+str(rho)+"_"+str(t)+"_"+str(p)+"_"+str(5)+grad_ext
						print(store_name)

						f = open(store_name, 'rb')
						grad_sparse = pickle.load(f)
						f.close()

						# Converting sparse gradient to full matrix gradient
						grad = sparse2grad(grad_sparse)

						grad_captures.append([(rho, t, p), grad])


				sim_pairs = []
				already_used = []
				previous_pair_0 = -1

				for pair in combinations(range(0, step*step), 2):

					if pair[0] not in already_used:

						if pair[0] != previous_pair_0: # new pair[0] -> new tree

							# Current tree creation (Root pair[0])
							act_tree = Tree()
							tree_list_0.append(act_tree) # passing by reference
							root_name = pos2name(grad_captures[pair[0]][0])

							act_tree.create_node(root_name, root_name, data = grad_captures[pair[0]][0])

							# Default child: pair[0]
							#act_tree.create_node(root_name, root_name, parent = root_name, data = grad_captures[pair[0]][0])

							previous_pair_0 = pair[0]

						if pair[1] not in already_used:

							grad1 = grad_captures[pair[0]] [1]
							grad2 = grad_captures[pair[1]] [1]

							#cv.imshow("Scanned gradient", abs(grad1[:, :, 0] - grad2[:, :, 0]))
							#cv.imshow("Current gradient", grad2[:, :, 0])
							#cv.waitKey(1)

							sim = similarity_vec(grad1, grad2)
							#sim = similarity_sparse(grad1, grad2)
							print(pair, sim)


							if sim > sim_thresh:
								# Adding children to root node
								child_name = pos2name(grad_captures[pair[1]][0])
								act_tree.create_node(child_name, child_name, parent = root_name, data = grad_captures[pair[1]][0])

								sim_pairs.append( (grad_captures[pair[0]][0], grad_captures[pair[1]][0], sim) )
								print("Added one")

								already_used.append(pair[1])

								# Adding full act_tree to tree_list level 0
								#act_tree.show()

							else :

								cv.imshow("Different enough gradients", abs(grad1[:, :, 0] - grad2[:, :, 0]))
								#cv.imshow("Current gradient", grad2[:, :, 0])
								cv.waitKey(1)


				# for tree in tree_list_0:
				# 	tree.show()

				#sys.exit()


				print(rho, theta, phi, "Len sim_pairs:", len(sim_pairs))
				print("Time", time.time() - t1)


	f = open(data_path + "tree_list_0" + grad_ext, 'wb')
	pickle.dump(tree_list_0, f)
	f.close()



def gen_lvl_others(mask_matrix, level, step, sim_thresh, pas):


	rg = np.arange(0, int(step*pas), pas)
	print(rg)
	pairs = np.linspace((0, 0), (int(step*pas), int(step*pas)), int(pas))

	tree_list_act = []


	for rho in range_rho:
		for theta in range_theta[0:-1:step]:
			for phi in range_phi[0:-1:step]:

				t1 = time.time()

				# List of saved gradients [(r, t, p), grad] 
				grad_captures = []

				# Importing gradients files for each position in the range
				for i in range(step):
					for j in range(step):

						t = theta + rg[i]
						p = phi + rg[j]

						r_i, t_i, p_i = pos2indexes(rho, t, p)

						if mask_matrix[r_i, t_i, p_i] == True:

							store_name = data_path + grad_ixt+obj_file_name+"_"+str(rho)+"_"+str(t)+"_"+str(p)+"_"+str(5)+grad_ext
							print(store_name)

							f = open(store_name, 'rb')
							grad_sparse = pickle.load(f)
							f.close()

							# Converting sparse gradient to full matrix gradient
							grad = sparse2grad(grad_sparse)

							grad_captures.append([(rho, t, p), grad])

						else:
							grad_captures.append(None)


				sim_pairs = []
				already_used = []
				previous_pair_0 = -1

				for pair in combinations(range(0, step*step), 2):

					if pair[0] not in already_used and grad_captures[pair[0]] != None:

						if pair[0] != previous_pair_0: # new pair[0] -> new tree

							# Current tree creation (Root pair[0])
							act_tree = Tree()
							tree_list_act.append(act_tree) # passing by reference
							root_name = pos2name(grad_captures[pair[0]][0])

							act_tree.create_node(root_name, root_name, data = grad_captures[pair[0]][0])

							# Default child: pair[0]
							#act_tree.create_node(root_name, root_name, parent = root_name, data = grad_captures[pair[0]][0])

							previous_pair_0 = pair[0]

						if pair[1] not in already_used and grad_captures[pair[1]] != None:

							grad1 = grad_captures[pair[0]] [1]
							grad2 = grad_captures[pair[1]] [1]

							#cv.imshow("Scanned gradient", abs(grad1[:, :, 0] - grad2[:, :, 0]))
							#cv.imshow("Current gradient", grad2[:, :, 0])
							#cv.waitKey(1)

							sim = similarity_vec(grad1, grad2)
							#sim = similarity_sparse(grad1, grad2)
							print(pair, sim)


							if sim > sim_thresh:
								# Adding children to root node
								child_name = pos2name(grad_captures[pair[1]][0])
								act_tree.create_node(child_name, child_name, parent = root_name, data = grad_captures[pair[1]][0])

								sim_pairs.append( (grad_captures[pair[0]][0], grad_captures[pair[1]][0], sim) )
								print("Added one")

								already_used.append(pair[1])

								# Adding full act_tree to tree_list level 0
								#act_tree.show()

							else :

								cv.imshow("Different enough gradients", abs(grad1[:, :, 0] - grad2[:, :, 0]))
								#cv.imshow("Current gradient", grad2[:, :, 0])
								cv.waitKey(1)


				#for tree in tree_list_act:
				#	tree.show()

				#sys.exit()


				print(rho, theta, phi, "Len sim_pairs:", len(sim_pairs))
				print("Time", time.time() - t1)


	f = open(data_path + "tree_list_" + str(level) + grad_ext, 'wb')
	pickle.dump(tree_list_act, f)
	f.close()



'''
For level 1+:

Load previous level file with trees
Construct mask matrix of trees positions : 1 if root tree exists here, 0 else
Iterate over matrix, with extended neighbours ranges

For each group:
	Load gradients
	Reduce grad size (possible?) or similarity threshold
	Pair similar gradients

-> New tree_list
Save as file



Size : 5 * 51 * 51 = 13005 grads
Level 0 (0.90) : 5 -> 5 * 10 * 10 = 500 min, 13k max, 1.3k avg
Level 1 :  10 -> 5 * 5 * 5 = 125 min


'''

def pos2indexes(r, t, p):
	r_i = np.where(range_rho == int(r))
	t_i = np.where(range_theta == int(t))
	p_i = np.where(range_phi == int(p))
	
	return r_i, t_i, p_i


def treelist2mask(tree_list):
	mask_matrix = np.zeros((len(range_rho), len(range_theta), len(range_phi)), dtype = "bool")

	for tree in tree_list:
		root_node = tree.get_node(tree.root)
		#print(root_node)

		r, t, p = name2pos(root_node.tag)
		r_i, t_i, p_i = pos2indexes(r, t, p)

		#print(r_i, t_i, p_i)
		mask_matrix[r_i, t_i, p_i] = True

	return mask_matrix


def render_mask(mask_matrix):
	for rho in range(mask_matrix.shape[0]):
		cv.imshow("mask_matrix rho " + str(rho), imutils.resize(mask_matrix[rho, :, :].astype("B")*255, width=mask_matrix.shape[1]*6))
		cv.waitKey(1)
	cv.waitKey(0)



pyramid = None


angle_threshold = np.radians(5.0)
print("Lim", angle_threshold)

sim_thresh = 0.85
resize_ratio = 0.8

step = 7
pas = range_theta[1] - range_theta[0] #2



#gen_lvl_0(5, 0.90, pas)


f = open(data_path + "tree_list_0" + grad_ext, 'rb')
tree_list_0 = pickle.load(f)
f.close()

print(len(tree_list_0))

mask_matrix_0 = treelist2mask(tree_list_0)
render_mask(mask_matrix_0)

#gen_lvl_others(mask_matrix_0, 1, 7, 0.85, pas)


f = open(data_path + "tree_list_1" + grad_ext, 'rb')
tree_list_1 = pickle.load(f)
f.close()

print(len(tree_list_1))

mask_matrix_1 = treelist2mask(tree_list_1)
render_mask(mask_matrix_1)

#gen_lvl_others(mask_matrix_1, 2, 10, 0.80, pas)


f = open(data_path + "tree_list_2" + grad_ext, 'rb')
tree_list_2 = pickle.load(f)
f.close()

print(len(tree_list_2))

mask_matrix_2 = treelist2mask(tree_list_2)
render_mask(mask_matrix_2)

#gen_lvl_others(mask_matrix_2, 3, 25, 0.80, pas)


f = open(data_path + "tree_list_3" + grad_ext, 'rb')
tree_list_3 = pickle.load(f)
f.close()

print(len(tree_list_3))

mask_matrix_3 = treelist2mask(tree_list_3)
render_mask(mask_matrix_3)




""" #TESTS

capture_name1 = data_path + capture_ixt+obj_file_name+"_"+str(7.0)+"_"+str(-50.0)+"_"+str(-20.0)+".png"
capture_name2 = data_path + capture_ixt+obj_file_name+"_"+str(7.0)+"_"+str(-50.0)+"_"+str(-24.0)+".png"

capture_name1 = data_path + "test7.png"
capture_name2 = data_path + "test1.png"


img1 = cv.imread(capture_name1)
img2 = cv.imread(capture_name2)

grad = gray2grad(cv.cvtColor(img1, cv.COLOR_BGR2GRAY))
color_grad = color2grad(img2)


angles = []
sim_plt = []

for a in range(-45, 45):
	grad_rot = imutils.rotate(color_grad, a)	
	
	#grad_render(img2, color_grad, (0, 0, 255), arrow_len = 30, step = 2)

	sim = similarity_vec(grad, grad_rot)
	print("Similarity", a, sim)

	angles.append(a)
	sim_plt.append(sim)

	cv.imshow("Grad rot", grad_rot[:,:,0])
	cv.waitKey(1)

plt.rcParams["figure.figsize"] = (8,6)
plt.plot(angles, sim_plt)
plt.xlabel('Angle')
plt.ylabel('similarity')
plt.show()

#cProfile.run('gray2grad(cv.cvtColor(img2, cv.COLOR_BGR2GRAY))')


#t1 = time.time()
#cart1 = grad2cart(grad)
#cart2 = grad2cart(color_grad)
#t1 = time.time()

#print("rotate")
#color_grad = imutils.rotate(color_grad, 90)


print("Similarity", similarity_vec(grad, color_grad))
#print("Time =", time.time() - t1)

#cProfile.run('similarity_vec(grad, color_grad)')



cv.imshow("Display 1", img1)
cv.imshow("Display 2", img2)
cv.imshow("gray2grad", grad[:, :, 0])
cv.imshow("color2grad", color_grad[:, :, 0])
print(np.max(color_grad))


cv.waitKey(0)

edge1 = image2edge(img1, angle_threshold)
edge2 = image2edge(img2, angle_threshold)

#cProfile.run('image2edge(img2, angle_threshold)')

cv.imshow("Color edge algorithm", edge2)
cv.waitKey(0)



for a in range(-10, 10):

	rotated = imutils.rotate(edge2, a)

	cv.imshow("Edge 1", rotated)
	#cv.imshow("Edge 2", rotated)
	cv.imshow("Difference", np.abs(rotated - color_grad))

	#print("Similarity", similarity(grad, cart2))

	#print(cosine_similarity(edge1, rotated))
	#print(compare_ssim(edge1, rotated))


	cv.waitKey(0)


cv.waitKey(0)




#kernel = np.ones((3, 3), np.uint8)
#edge_eroded = cv.erode(edge, kernel)
#cv.imshow("Eroded", edge_eroded)
#cv.waitKey(0)

"""




			
"""
			for pair1 in combinations(range(0, int(step*pas), int(pas)), 2):

				t1 = theta + pair1[0]
				p1 = phi + pair1[1]

				capture_name1 = data_path + capture_ixt+obj_file_name+"_"+str(rho)+"_"+str(t1)+"_"+str(p1)+".png"
				img1 = cv.imread(capture_name1, cv.IMREAD_GRAYSCALE)
				grad1 = gray2grad(img1)

				for pair2 in combinations(range(pair1, int(step*pas), int(pas)), 2):

					if pair1 != pair2 :

						t2 = theta + pair2[0]
						p2 = phi + pair2[1]

						capture_name2 = data_path + capture_ixt+obj_file_name+"_"+str(rho)+"_"+str(t2)+"_"+str(p2)+".png"
						img2 = cv.imread(capture_name2, cv.IMREAD_GRAYSCALE)
						grad2 = gray2grad(img2)


						sim = similarity_vec(grad1, grad2)

						sim_pairs.append(sim)
						print("Added one")
"""