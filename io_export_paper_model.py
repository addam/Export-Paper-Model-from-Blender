# -*- coding: utf-8 -*-
# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

bl_info = {
	"name": "Export Paper Model",
	"author": "Addam Dominec",
	"version": (0, 8),
	"blender": (2, 6, 4),
	"api": 44028,
	"location": "File > Export > Paper Model",
	"warning": "",
	"description": "Export printable net of the active mesh",
	"category": "Import-Export",
	"wiki_url": "http://wiki.blender.org/index.php/Extensions:2.6/Py/" \
		"Scripts/Import-Export/Paper_Model",
	"tracker_url": "https://projects.blender.org/tracker/index.php?" \
		"func=detail&aid=22417&group_id=153&atid=467"}

"""

Additional links:
    e-mail: adominec {at} gmail {dot} com

"""
import bpy
import mathutils as M
import mathutils.geometry as G
from heapq import heappush, heappop
from math import pi
priority_effect={
	'convex':0.5,
	'concave':1,
	'length':-0.05}
lines = list()
twisted_quads = list()
labels = dict()
highlight_faces = list()

strf="{:.3f}".format

def sign(a):
	"""Return -1 for negative numbers, 1 for positive and 0 for zero."""
	if a == 0:
		return 0
	if a < 0:
		return -1
	else:
		return 1

def vectavg(vectlist):
	"""Vector average of given list."""
	if len(vectlist)==0:
		return M.Vector((0,0))
	last=vectlist[0]
	if type(last) is Vertex:
		vect_sum=last.co.copy() #keep the dimensions
		vect_sum.zero()
		for vect in vectlist:
			vect_sum+=vect.co
	else:
		vect_sum=last.copy() #keep the dimensions
		vect_sum.zero()
		for vect in vectlist:
			vect_sum+=vect
	return vect_sum/vectlist.__len__()

def angle2d(direction, unit_vector=M.Vector((1,0))):
	"""Get the view angle of point from origin."""
	if direction.length == 0.0:
		raise ValueError("Zero vector does not define an angle")
	if len(direction) >= 3: #for 3d vectors
		direction=direction.copy()
		direction.resize_2d()
	angle=direction.angle(unit_vector)
	if direction[1]<0:
		angle=2*pi-angle #hack for angles greater than pi
	return angle

def pairs(sequence):
	"""Generate consequent pairs throughout the given sequence; at last, it gives elements last, first."""
	i=iter(sequence)
	previous=first=next(i)
	for this in i:
		yield previous, this
		previous=this
	yield this, first

def fitting_matrix(v1, v2):
	"""Matrix that rotates v1 to the same direction as v2"""
	return (1/v1.length_squared)*M.Matrix((
		(+v1.x*v2.x +v1.y*v2.y, +v1.x*v2.y -v1.y*v2.x),
		(+v1.y*v2.x -v1.x*v2.y, +v1.x*v2.x +v1.y*v2.y)))

def z_up_matrix(n):
	"""Get a rotation matrix that aligns given vector upwards."""
	b=n.xy.length
	l=n.length
	if b>0:
		return M.Matrix((
			(n.x*n.z/(b*l),	-n.y/b, 0),
			(n.y*n.z/(b*l),  n.x/b, 0),
			(         -b/l,      0, 0)))
	else: #no need for rotation
		return M.Matrix((
			(1,	        0, 0),
			(0,	sign(n.z), 0),
			(0,         0, 0)))

class UnfoldError(ValueError):
	pass

class Unfolder:
	def __init__(self, ob):
		self.ob=ob
		self.mesh=Mesh(ob.data, ob.matrix_world)

	def prepare(self, properties=None):
		"""Something that should be part of the constructor - TODO """
		self.mesh.cut_obvious()
		self.mesh.generate_cuts()
		self.mesh.finalize_islands()
		self.mesh.save_uv()

	def save(self, properties):
		"""Export the document."""
		filepath=properties.filepath
		if filepath[-4:]==".svg" or filepath[-4:]==".png":
			filepath=filepath[0:-4]
		page_size = M.Vector((properties.output_size_x, properties.output_size_y)) #real page size in meters FIXME: must be scaled according to unit settings ?
		scale = bpy.context.scene.unit_settings.scale_length * properties.model_scale
		ppm = properties.output_dpi * 100 / 2.54 #points per meter
		self.mesh.generate_stickers(default_width = properties.sticker_width * page_size.y / scale)
		#Scale everything so that page height is 1
		self.mesh.finalize_islands(scale_factor = scale / page_size.y)
		self.mesh.fit_islands(aspect_ratio = page_size.x / page_size.y)
		if not properties.output_pure:
			self.mesh.save_uv(aspect_ratio = page_size.x / page_size.y)
			#TODO: do we really need a switch of our own?
			selected_to_active = bpy.context.scene.render.use_bake_selected_to_active; bpy.context.scene.render.use_bake_selected_to_active = properties.bake_selected_to_active
			self.mesh.save_image(filepath, page_size * ppm)
			#revoke settings
			bpy.context.scene.render.use_bake_selected_to_active=selected_to_active
		svg=SVG(page_size * ppm, properties.output_pure)
		svg.add_mesh(self.mesh)
		svg.write(filepath)

class Mesh:
	"""Wrapper for Bpy Mesh"""
	
	def __init__(self, mesh, matrix):
		global lines, twisted_quads
		self.verts=dict()
		self.edges=dict()
		self.edges_by_verts_indices=dict()
		self.faces=dict()
		self.islands=list()
		self.data=mesh
		self.pages=list()
		for bpy_vertex in mesh.vertices:
			self.verts[bpy_vertex.index]=Vertex(bpy_vertex, self, matrix)
		for bpy_edge in mesh.edges:
			edge=Edge(bpy_edge, self, matrix)
			self.edges[bpy_edge.index] = edge
			self.edges_by_verts_indices[(edge.va.index, edge.vb.index)] = edge
			self.edges_by_verts_indices[(edge.vb.index, edge.va.index)] = edge
		for bpy_face in mesh.faces:
			face = Face(bpy_face, self)
			self.faces[bpy_face.index] = face
		for index in self.edges:
			self.edges[index].process_faces()
	
	def cut_obvious(self):
		"""Cut all seams and non-manifold edges."""
		count=0
		for i in self.edges:
			edge = self.edges[i]
			if not edge.is_cut() and (edge.data.use_seam or len(edge.faces)<2): #NOTE: Here is one of the rare cases when using the original BPy data
				edge.cut()
				count += 1
	
	def generate_cuts(self):
		"""Cut the mesh so that it will be unfoldable."""
		global twisted_quads, labels
		twisted_quads = list()
		labels = dict()
		#Silently check that all quads are flat
		for index in self.faces:
			self.faces[index].check_twisted()
		# check for edges that are cut permanently
		edges = list()
		edges = [self.edges[edge_id] for edge_id in self.edges if not self.edges[edge_id].is_main_cut]
		if not edges:
			return True
		for edge in edges:
			edge.is_main_cut = True
		
		global differences
		differences = list()

		average_length = sum(edge.length for edge in edges) / len(edges)
		for edge in edges:
			edge.generate_priority(average_length)
		edges.sort(key = lambda edge:edge.priority, reverse=False)
		self.islands = set(map(Island, self.faces.values()))
		for edge in edges:
			face_a, face_b = edge.faces[:2]
			island_a, island_b = face_a.uvface.island, face_b.uvface.island
			if len(island_b.faces) > len(island_a.faces):
				island_a, island_b = island_b, island_a
			if island_a is not island_b:
				if island_a.join(island_b, edge):
					self.islands.remove(island_b)
		differences.sort(reverse=True)
		if differences[0][0]>1+1e-5:
			print ("""Papermodel warning: there are non-flat faces, which will be deformed in the output image.
			Showing first five values (normally they should be very close to 1.0):""")
			for diff in differences[0:5]:
				print ("{:.5f}".format(diff[0]))
		return True
		
	def generate_islands(self):
		"""DELETE ME Divide faces into several Islands."""
		def connected_faces(border_edges, inner_faces):
			outer_faces=list()
			for edge in border_edges:
				for face in edge.faces:
					if face not in inner_faces and face not in outer_faces:
						outer_faces.append(face)
			next_border=list()
			for face in outer_faces:
				for edge in face.edges:
					if not edge in border_edges:
						next_border.append(edge)
			if len(next_border)>0:
				outer_faces.extend(connected_faces(next_border, outer_faces))
			return outer_faces
		self.islands=list()
		remaining_faces=list(self.faces.values())
		#DEBUG: checking the transformation went ok
		global differences
		differences = list()
		while len(remaining_faces) > 0:
			self.islands.append(Island(remaining_faces))
		differences.sort(reverse=True)
		if differences[0][0]>1+1e-5:
			print ("""Papermodel warning: there are non-flat faces, which will be deformed in the output image.
			Showing first five values (normally they should be very close to 1.0):""")
			for diff in differences[0:5]:
				print ("{:.5f}".format(diff[0]))
		#FIXME: Why this?
		for edge in self.edges.values():
			edge.uvedges.sort(key=lambda uvedge: edge.faces.index(uvedge.uvface.face))
	
	def generate_stickers(self, default_width):
		"""Add sticker faces where they are needed."""
		#TODO: it should take into account overlaps with faces and with already created stickers and size of the face that sticker will be actually sticked on
		def uvedge_priority(uvedge):
			"""Retuns whether it is a good idea to create a sticker on this edge"""
			#This is just a placeholder
			return uvedge.va.co.y
		for edge in self.edges.values():
			if edge.is_cut() and len(edge.uvedges) >= 2:
				if uvedge_priority(edge.uvedges[0]) >= uvedge_priority(edge.uvedges[1]):
					edge.uvedges[0].island.add(Sticker(edge.uvedges[0], default_width))
				else:
					edge.uvedges[1].island.add(Sticker(edge.uvedges[1], default_width))
			if len(edge.uvedges) > 2:
				for additional_uvedge in edge.uvedges[2:]:
					additional_uvedge.island.add(Sticker(additional_uvedge, default_width))
	
	def finalize_islands(self, scale_factor=1):
		for island in self.islands:
			island.apply_scale(scale_factor)
			island.generate_bounding_box()
	
	def largest_island_ratio(self, page_size):
		largest_ratio=0
		for island in self.islands:
			ratio=max(island.bounding_box.x/page_size.x, island.bounding_box.y/page_size.y)
			largest_ratio=max(ratio, largest_ratio)
		return largest_ratio
	
	def fit_islands(self, aspect_ratio):
		"""Move islands so that they fit into pages, based on their bounding boxes"""
		#this algorithm is not optimal, but cool enough
		#it handles with two basic domains:
		#list of Points: they describe all sensible free rectangle area available on the page (one rectangle per Point)
		#Boundaries: linked list of points around the used area and the page - makes many calculations a lot easier
		page_size=M.Vector((aspect_ratio, 1))
		class Boundary:
			"""Generic point in usable boundary defined by rectangles and borders of the page"""
			def __init__(self, x, y):
				self.previous=None
				self.next=None
				self.point=None
				self.x=x
				self.y=y
			def __rshift__(self, next):
				"""Connect another Boundary after this one."""
				if next and next.next and ((next.x<self.x and next.next.x>next.x) or (next.y<self.y and next.next.y>next.y)):
					self >> next.next
				elif next is not self:
					self.next=next
					if next:
						next.previous=self
				return self
		class Point:
			"""A point on the boundary we may want to attach a rectangle to"""
			def __init__(self, boundary, points):
				self.points=points
				points.append(self)
				self.boundary=boundary
				boundary.point=self
				self.previous=boundary.previous
				self.next=boundary.next
				#if we are an outer corner...
				if boundary.previous and boundary.previous.y<boundary.y:
					current=self.previous
					while current: #go along the boundary to left up
						if current.y > boundary.y:
							self.boundary=Boundary(current.x, boundary.y)
							current.previous >> self.boundary
							self.boundary >> current
							break
						current=current.previous
				elif boundary.next and boundary.next.x < boundary.x:
					current=self.next
					while current: #go along the boundary to right down
						if current.x > boundary.y:
							self.boundary=Boundary(boundary.x, current.y)
							current.previous >> self.boundary
							self.boundary >> current
							break
						current=current.next
				#calculate the maximal rectangle that can fit here
				self.area=M.Vector((0,0))
				current=self.previous
				while current: #go along the boundary to left up
					if current.x > self.boundary.x:
						self.area.y=current.y-self.boundary.y
						break
					current=current.previous
				current=self.next
				while current: #go along the boundary to right down
					if current.y > self.boundary.y:
						self.area.x=current.x-self.boundary.x
						break
					current=current.next
				self.niceness=self.area.x*self.area.y
			def add_island(self, island):
				"""Attach an island to this point and update all affected neighbourhood."""
				island.pos=M.Vector((self.boundary.x, self.boundary.y))
				#you have to draw this if you want to get it
				a=Boundary(island.pos.x, island.pos.y+island.bounding_box.y)
				b=Boundary(island.pos.x+island.bounding_box.x, island.pos.y+island.bounding_box.y)
				c=Boundary(island.pos.x+island.bounding_box.x, island.pos.y)
				a >> b
				b >> c
				if self.previous:
					self.previous.next >> self.next.previous #note: they don't have to be the same. They are rather the closest obstacles
					self.previous >> a
				c >> self.next.previous
				Point(a, self.points)
				Point(c, self.points)
				#update all points whose area overlaps this island
				current=self.previous
				while current and current.x<self.boundary.x:
					if current.point:
						if current.point.area.x+current.x > self.boundary.x:
							current.point.area.x=self.boundary.x-current.x
					current=current.previous
				current=self.next
				while current and current.y<self.boundary.y:
					if current.point:
						if current.point.area.y+current.y > self.boundary.y:
							current.point.area.y=self.boundary.y-current.y
					current=current.next
				#If we have no reference to the rest of the boundary, let us rest in peace
				if (not self.previous or self.previous.next is not self) and \
					(not self.next or self.next.previous is not self):
						self.points.remove(self)
			def __str__(self):
				return "Point at "+str(self.boundary.x)+" "+str(self.boundary.y)+" of available area "+str(self.area)
		#fixme: at first, it should cut all islands that are too big to fit the page
		#todo: there should be a list of points off the boundary that are created from pairs of open edges
		largest_island_ratio = self.largest_island_ratio(page_size) 
		if largest_island_ratio > 1:
			raise UnfoldError("An island is too big to fit to the page size. To make the export possible, scale the object down "+strf(largest_island_ratio)+" times.")
		islands=list(self.islands)
		#sort islands by their ugliness (we need an ugly expression to treat ugliness correctly)
		islands.sort(key=lambda island: (lambda vector:-pow(vector.x, 2)-pow(vector.y, 2)-pow(vector.x-vector.y, 2))(island.bounding_box))
		remaining_count=len(islands)
		page_num=1
		while remaining_count > 0:
			#create a new page and try to fit as many islands onto it as possible
			page=Page(page_num)
			page_num+=1
			points=list()
			#We start with the whole page
			a=Boundary(page_size.x, page_size.y)
			b=Boundary(0, page_size.y)
			c=Boundary(0, 0)
			d=Boundary(page_size.x, 0)
			e=Boundary(page_size.x, page_size.y)
			a >> b
			b >> c
			c >> d
			d >> e
			Point(c, points)
			for island in islands:
				if not island.is_placed:
					for point in points:
						#test if it would fit to this point
						if island.bounding_box.x<=point.area.x and island.bounding_box.y<=point.area.y:
							point.add_island(island)
							island.is_placed=True
							page.add(island)
							remaining_count -= 1
							break
					points.sort(key=lambda point: point.niceness) #ugly points first (to get rid of them)
			self.pages.append(page)
	
	def save_uv(self, aspect_ratio=1): #page_size is in pixels
		bpy.ops.object.mode_set()
		bpy.ops.mesh.uv_texture_add()
		#note: expecting that the active object's data is self.mesh
		tex=self.data.uv_textures.active
		tex.name="Unfolded"
		for island in self.islands:
			island.save_uv(tex, aspect_ratio)
	
	def save_image(self, filename, page_size_pixels:M.Vector):
		rd=bpy.context.scene.render
		recall_margin=rd.bake_margin; rd.bake_margin=0
		recall_clear=rd.use_bake_clear; rd.use_bake_clear=False
		for page in self.pages:
			#image=bpy.data.images.new(name="Unfolded "+self.data.name+" "+page.name, width=int(page_size.x), height=int(page_size.y))
			image_name=(self.data.name[:16]+" "+page.name+" Unfolded")[:20]
			obstacle=bpy.data.images.get(image_name)
			if obstacle:
				obstacle.name=image_name[0:-1] #when we create the new image, we want it to have *exactly* the name we assign
			bpy.ops.image.new(name=image_name, width=int(page_size_pixels.x), height=int(page_size_pixels.y), color=(1,1,1,1))
			image=bpy.data.images.get(image_name) #this time it is our new image
			if not image:
				print ("papermodel ERROR: could not get image", image_name)
			image.filepath_raw=filename+"_"+page.name+".png"
			image.file_format='PNG'
			texfaces=self.data.uv_textures.active.data
			for island in page.islands:
				for uvface in island.faces:
					if not uvface.is_sticker:
						texfaces[uvface.face.index].image=image
			bpy.ops.object.bake_image()
			image.save()
			for island in page.islands:
				for uvface in island.faces:
					if not uvface.is_sticker:
						texfaces[uvface.face.index].image=None
			image.user_clear()
			bpy.data.images.remove(image)
		rd.bake_margin=recall_margin
		rd.use_bake_clear=recall_clear
   
class Vertex:
	"""BPy Vertex wrapper"""
	
	def __init__(self, bpy_vertex, mesh=None, matrix=1):
		self.data=bpy_vertex
		self.index=bpy_vertex.index
		self.co=bpy_vertex.co*matrix
		self.edges=list()
		self.uvs=list()
	
	def __hash__(self):
		return hash(self.index)
	def __eq__(self, other):
		if type(other) is type(self):
			return self.index==other.index
		else:
			return False
	def __sub__(self, other):
		return self.co-other.co
	def __rsub__(self, other):
		if type(other) is type(self):
			return other.co-self.co
		else:
			return other-self.co
	def __add__(self, other):
		return self.co+other.co
	def __str__(self):
		return "Vertex {} at: {}".format(self.index, self.co[0:2])
	def __repr__(self):
		return "Vertex(id={}...)".format(self.index)
		
class Edge:
	"""Wrapper for BPy Edge"""
	
	def __init__(self, edge, mesh, matrix=1):
		self.data=edge
		self.va=mesh.verts[edge.vertices[0]]	
		self.vb=mesh.verts[edge.vertices[1]]
		self.vect=self.vb.co-self.va.co
		self.length=self.vect.length
		self.faces=list()
		self.angles=dict()
		self.other_face=dict() #FIXME: this should be rather a function, damn this mess
		self.uvedges=list()

		self.is_main_cut=False #defines whether the first two faces are connected; all the others will be automatically treated as cut
		self.priority=None
		self.va.edges.append(self)
		self.vb.edges.append(self)
	
	def process_faces(self):
		"""Reorder faces (if more than two), calculate angle(s) and if normals are wrong, mark edge as cut"""
		def niceness (angle):
			"""Return how good idea it would be to leave the given angle uncut"""
			if angle<0:
				if priority_effect['concave']!=0:
					return -1/(angle*priority_effect['concave'])
			elif angle>0:
				if priority_effect['convex']!=0:
					return 1/(angle*priority_effect['convex'])
			#if angle == 0:
			return 1000000
		if len(self.faces)==0:
			return
		elif len(self.faces)==1:
			self.angles[self.faces[0]]=pi
			return
		else:
			#hacks for edges with two or more faces connected
			rot=z_up_matrix(self.vect) #Everything is easier in 2D
			normal_directions=dict() #direction of each face's normal, rotated to 2D
			face_directions=dict() #direction which each face is pointing in from this edge; rotated to 2D
			is_normal_cw=dict()
			for face in self.faces:
				#DEBUG
				if (face.normal*rot).z > 1e-4:
					print ("papermodel ERROR in geometry, deformed face:", face.normal*rot)
				try:
					normal_directions[face]=angle2d((face.normal*rot).xy)
					face_directions[face] = angle2d(((vectavg(face.verts)-self.va.co)*rot).xy)
				except ValueError:
					raise UnfoldError("Fatal error: there is a face with two edges in the same direction.")
				is_normal_cw[face] = (normal_directions[face] - face_directions[face]) % (2*pi) < pi #True for clockwise normal around this edge, False for ccw
			#Firstly, find which two faces will be the 'main' ones
			self.faces.sort(key=lambda face: normal_directions[face])
			best_pair = 0, None, None #tuple: niceness, face #1, face #2
			for first_face, second_face in pairs(self.faces):
				if is_normal_cw[first_face] != is_normal_cw[second_face]:
					used = True
					#Always calculate the inner angle (between faces' backsides)
					if not is_normal_cw[first_face]:
						first_face, second_face = second_face, first_face
					#Get the angle difference
					angle_normals=(normal_directions[second_face]-normal_directions[face]) % (2*pi)
					#Check whether it is better than the current best one
					if niceness(angle_normals) > best_pair[0]:
						best_pair=niceness(angle_normals), first_face, second_face
			#For each face, find the nearest neighbour from its backside
			for index, face in enumerate(sorted(self.faces, key=lambda face: face_directions[face])):
				if is_normal_cw[face]:
					adjacent_face=self.faces[(index-1) % len(self.faces)]
				else:
					adjacent_face=self.faces[(index+1) % len(self.faces)]
				self.other_face[face]=adjacent_face
			#Overwrite the calculated neighbours for the two 'main' ones
			if best_pair[0] > 0:
				#If we found two nice faces, create a connection between them
				for first_face, second_face in pairs ([best_pair[1], best_pair[2]]):
					self.other_face[first_face]=second_face
				#Reorder the list of faces so that the main ones come first
				index_first=self.faces.index(best_pair[1])
				self.faces=self.faces[index_first:] + self.faces[:index_first]
			else:
				#If none of the faces is nice, go cut yourself
				self.cut() 
			#Calculate angles for each face to the 'other face' (self.other_face[face])
			for face in self.faces:
				angle_faces = pi - ((face_directions[self.other_face[face]] - face_directions[face]) % (2*pi))
				#Always calculate the inner angle (between faces' backsides)
				if is_normal_cw[face]:
					angle_faces = -angle_faces
				self.angles[face] = angle_faces
	
	def generate_priority(self, average_length=1):
		"""Calculate initial priority value."""
		angle = self.angles[self.faces[0]]
		if angle > 0:
			self.priority = (angle/pi)*priority_effect['convex']
		else:
			self.priority = -(angle/pi)*priority_effect['concave']
		length_effect = (self.length/average_length) * priority_effect['length']
		self.priority += length_effect
		labels[self] = [(self.va+self.vb)*0.5, strf(self.priority)]
	
	def is_cut(self, face=None):
		"""Optional argument 'face' defines who is asking (useful for edges with more than two faces connected)"""
		#Return whether there is a cut between the two main faces
		if face is None or self.faces.index(face) <= 1:
			return self.is_main_cut
		#All other faces (third and more) are automatically treated as cut
		else:
			return True
	
	def label_update(self):
		"""Debug tool"""
		global labels
		labels[self][1] = strf(self.priority)
	
	def cut(self):
		"""Set this edge as cut."""
		self.data.use_seam=True #TODO: this should be optional; NOTE: Here, the original BPy Edge data is used
		self.is_main_cut=True
		if self.priority:
			self.label_update()

	def __lt__(self, other):
		"""Compare by priority."""
		return self.priority < other.priority
	def __gt__(self, other):
		"""Compare by priority."""
		return self.priority > other.priority
	def __str__(self):
		return "Edge id: {}".format(self.data.index)
	def __repr__(self):
		return "Edge(id={}...)".format(self.data.index)
	def other_vertex(self, this):
		"""Get a vertex of this edge that is not the given one - or None if none of both vertices is the given one."""
		if self.va is this:
			return self.vb
		elif self.vb is this:
			return self.va
		return None
	def other_uvedge(self, this):
		"""Get an uvedge of this edge that is not the given one - or None if no other uvedge was found."""
		for uvedge in self.uvedges:
			if uvedge is not this:
				return uvedge
		else:
			return None

class Face:
	"""Wrapper for BPy Face"""
	def __init__(self, bpy_face, mesh, matrix=1):
		self.data = bpy_face
		self.index = bpy_face.index
		self.edges = list()
		self.verts = [mesh.verts[i] for i in bpy_face.vertices]
		#TODO: would be nice to reuse the existing normal if possible
		self.normal = (self.verts[1]-self.verts[0]).cross(self.verts[2]-self.verts[0]).normalized()
		for verts_indices in bpy_face.edge_keys:
			edge = mesh.edges_by_verts_indices[verts_indices]
			self.edges.append(edge)
			edge.faces.append(self)
	def check_twisted(self):
		if len(self.verts) > 3:
			global twisted_quads, lines
			vert_a=self.verts[0]
			normals=list()
			for vert_b, vert_c in zip(self.verts[1:-1], self.verts[2: ]):
				normal=(vert_b.co-vert_a.co).cross(vert_c.co-vert_b.co)
				normal /= (vert_b.co-vert_a.co).length * (vert_c.co-vert_b.co).length #parallel edges have lesser weight, but length does not have an effect
				normals.append(normal)
				lines.append(((vert_b.co+vert_c.co)/2, (vert_b.co+vert_c.co)/2+normal.copy().normalized()))
			average_normal=vectavg(normals)
			for normal in normals:
				if normal.angle(average_normal) > 0.01: #TODO: this threshold should be editable or well chosen at least
					twisted_quads.append(self.verts)
					return True
		return False
	def __hash__(self):
		return hash(self.index)
	def __str__(self):
		return "Face id: "+str(self.index)
	def __repr__(self):
		return "Face(id="+str(self.index)+"...)"

class Island:
	def __init__(self, face=None):
		"""Create an Island from a single Face"""
		self.faces=list()
		self.edges=list()
		self.verts=set()
		self.stickers=list()
		self.pos=M.Vector((0,0))
		self.offset=M.Vector((0,0))
		self.angle=0
		self.is_placed=False
		self.bounding_box=M.Vector((0,0))
		
		if face:
			self.add(UVFace(face, self))
		
		self.boundary_sorted = list(self.edges)
		self.boundary_sorted.sort()		

	def join(self, other, edge:Edge) -> bool:
		"""
		Try to join other island on given edge
		Returns False if they would overlap
		"""
		
		class Intersection(Exception):
			pass
			
		def is_below(self: UVEdge, other: UVEdge, verbose=False, epsilon=1e-20):
			#FIXME: estimate the epsilon based on input vectors
			if self is other: #or (self.va.co==other.vb.co and self.vb.co == other.va.co):
				return False
			if self.max <= other.min or self.min >= other.max:
				#DEBUG
				if verbose:
					print("compare {}, {}: Whatever.".format(self, other))
				return other.min < self.min
			cross_b1 = (self.max - self.min).cross(other.min - self.min).z
			cross_b2 = (self.max - self.min).cross(other.max - self.min).z
			if abs(cross_b1) < epsilon:
				cross_b1 = 0.0
			if abs(cross_b2) < epsilon:
				cross_b2 = 0.0
			if cross_b1 >= 0 and cross_b2 >= 0:
				#DEBUG
				if verbose:
					print("{} is 1 above {} ({}, {})".format(other, self, cross_b1, cross_b2))
				return True
			if cross_b1 <= 0 and cross_b2 <= 0:
				if verbose:
					print("{} is 2 above {} ({}, {}; {}, {})".format(self, other, cross_b1, cross_b2, self.min is other.max, self.max is other.min))
				return False
			cross_a1 = (other.max - other.min).cross(self.min - other.min).z
			cross_a2 = (other.max - other.min).cross(self.max - other.min).z
			if abs(cross_a1) < epsilon:
				cross_a1 = 0.0
			if abs(cross_a2) < epsilon:
				cross_a2 = 0.0
			if cross_a1 <= 0 and cross_a2 <= 0:
				if verbose:
					print("{} is 4 above {} ({}, {})".format(other, self, cross_a1, cross_a2))
				return True
			if cross_a1 >= 0 and cross_a2 >= 0:
				if verbose:
					print("{} is 3 above {} ({}, {})".format(self, other, cross_a1, cross_a2))
				return False
			if verbose:
				print ("{} intersects {}".format(self, other))
			raise Intersection()

		# DEBUG
		def is_below_lagrange(self: UVEdge, other:UVEdge, verbose = False):
			if self is other:
				return False
			if self.max <= other.min or self.min >= other.max:
				if self.min == other.min:
					print ("=====================================================", end="")
				print(".", end="")
				return other.min < self.min
			if self.min > other.min or other.max.co.x - other.min.co.x == 0:
				is_inverse = True
				self, other = other, self
			else:
				is_inverse = False
			if other.max.co.x - other.min.co.x == 0:
				return is_inverse if other.min < self.min else not is_inverse
			x, y = self.min.co[0:2] if self.min != other.min else self.max.co[0:2]
			value = (other.min.co.y * (other.max.co.x-x) + other.max.co.y * (x-other.min.co.x)) / (other.max.co.x - other.min.co.x)
			return is_inverse if value < y else not is_inverse
		
		class Sweepline:
			def __init__(self):
				self.root = TreeNode(list())
				self.count = 0
				#DEBUG
				self.history = list()
		
			def dump(self, force = False):
				children = list()
				stack = [self.root]
				while stack:
					node = stack.pop()
					if node.split_value:
						stack += node.children
					else:
						children = node.children + children
				if force:
					print("Forcing dump... {} children".format(len(children)))
				problems = 0
				for a, edge_a in enumerate(children[:-1]):
					for edge_b in children[a+1:]:
						try:
							if is_below(edge_b, edge_a) or not is_below(edge_a, edge_b):
								problems += 1
						except Intersection:
							problems += 1
				if not problems and not force:
					return
				for a, edge_a in enumerate(children):
					print ("{}: {}".format(a, edge_a))
				for a in range(len(children)):
					print(a, end=" ")
				print()
				for a, edge_a in enumerate(children):
					for b, edge_b in enumerate(children):
						if a < b:
							try:
								c1 = is_below(edge_a, edge_b)
								c2 = is_below(edge_b, edge_a)
								print("!" if c1 and c2 else "+" if c1 else "-" if c2 else "0", end=" ")
							except Intersection:
								print ("#", end="!")
								raise RuntimeError()
						else:
							print (" ", end=" ")
					print(a)
			
			def add(self, item, cmp = is_below):
				#DEBUG
				self.neighbor_check(item)
				
				node = self.root
				while node.split_value:
					node = node.children[0 if cmp(item, node.split_value) else 1]
				index = node.index(item)
				node.children.insert(index, item)
				
				"""
				if len(node.children) > TreeNode.max_children:
					print("Added", item)
					for a, edge_a in enumerate(node.children):
						print ("{}: {} min: {}".format(a, edge_a, edge_a.min))
					for a in range(len(node.children)):
						print (a, end="  ")
					print()
					for a, edge_a in enumerate(node.children):
						for b, edge_b in enumerate(node.children):
							if a<b:
								c1 = is_below_lagrange(edge_a, edge_b)
								c2 = is_below_lagrange(edge_b, edge_a)
								print ("!" if c1 and c2 else "+" if c1 else "-" if c2 else "0", end="")
								print ("1" if edge_a.va is edge_b.va else "2" if edge_a.va is edge_b.vb else "3" if edge_a.vb is edge_b.va else "4" if edge_a.vb is edge_b.vb else " ", end=" ")
							else:
								print ("   ", end="")
						print(a)
				"""
				if len(node.children) > TreeNode.max_children:
					split_index = TreeNode.max_children//2
					left = TreeNode(node.children[:split_index])
					right = TreeNode(node.children[split_index:])
					node.split_value = node.children[split_index]
					node.children = [left, right]
			
			def remove(self, item, cmp = is_below):
				#print("Remove {}".format(item))
				parent = None
				splitting_nodes = list()
				node = self.root
				while node.split_value:
					parent = node
					if item is node.split_value:
						splitting_nodes.append(node) # must replace with something else
						node = node.children[1]
					else:
						node = node.children[0 if cmp(item, node.split_value) else 1]
				#DEBUG
				index = node.children.index(item)
				if index > 0 and index < len(node.children)-1:
					if not cmp(node.children[index-1], node.children[index+1]):
						self.dump()
						raise RuntimeError()
				node.children.remove(item)
				if splitting_nodes:
					if node.children:
						# use the closest neighbour as a replacement
						split_value = node.children[0]
					else:
						# find a replacement elsewhere in the tree
						if node is parent.children[0]:
							split_value = parent.children[1]
							while type(split_value) is TreeNode:
								split_value = split_value.children[0]
						else:
							split_value = parent.children[0]
							while type(split_value) is TreeNode:
								split_value = split_value.children[-1]
					for splitting_node in splitting_nodes:
						splitting_node.split_value = split_value
				
				if parent and not node.children:
					print("(remove node l{}, r{})".format(len(parent.children[0].children), len(parent.children[1].children)), end=" ")
					parent.children.remove(node)
					parent.split_value = parent.children[0].split_value
					parent.children = parent.children[0].children
				elif parent and not parent.children[0].split_value and not parent.children[1].split_value and \
						len(parent.children[0].children) + len(parent.children[1].children) < TreeNode.max_children:
					parent.split_value = None
					print("(join nodes)", end=" ")
					parent.children = parent.children[0].children + parent.children[1].children
		
			def neighbor_check(self, item, cmp = is_below):
				left = self.root
				right = None
				while left.split_value:
					if cmp(item, left.split_value):
						right = left
						left = left.children[0]
					else:
						left = left.children[1]
				index = left.index(item)
				if index == len(left.children):
					debug = "Next node"
					parents = list()
					if right:
						parents.append(right.split_value)
						right = right.children[1]
						while type(right) is TreeNode:
							if right.split_value:
								debug += "+"
								parents.append(right.split_value)
							else:
								debug += "."
							right = right.children[0]
						for parent in parents:
							if not cmp(parent, right):
								debug += "!"
							else:
								debug += "~"
				else:
					debug = "This node @{}".format(index)
					right = left.children[index]
				left = left.children[index-1] if index > 0 else None
				#DEBUG
				if left and not cmp(left, item):
					print ("LEFT", left.va, left.vb, item.va, item.vb, cmp(left, item, True), cmp(item, left), is_below_lagrange(left, item), is_below_lagrange(item, left))
					self.dump(True)
					raise RuntimeError()
				#assert not left or cmp(left, item)
				if right and not cmp(item, right):
					print ("LEFT", left.va, left.vb, item.va, item.vb, cmp(left, item, True), cmp(item, left), is_below_lagrange(left, item), is_below_lagrange(item, left))
					print ("RIGHT", item.va, item.vb, right.va, right.vb, cmp(item, right, True), cmp(right, item), is_below_lagrange(item, right), is_below_lagrange(right, item))
					print(debug)
					self.dump(True)
					raise RuntimeError()
				#assert not right or cmp(item, right)
		
		class TreeNode:
			max_children = 8
			def __init__(self, seq):
				self.children = seq
				self.split_value = None
			def index(self, item, i = 0, cmp=is_below):
				#FIXME: bisect
				while i < len(self.children) and cmp(self.children[i], item):
					i += 1
				return i

		#find edge in other and in self
		for uvedge in edge.uvedges:
			if uvedge in self.edges:
				uvedge_a = uvedge
			elif uvedge in other.edges:
				uvedge_b = uvedge
		assert uvedge_a is not uvedge_b
		
		#determine rotation
		rot = fitting_matrix(uvedge_b.va - uvedge_b.vb, uvedge_a.vb - uvedge_a.va)
		trans = uvedge_a.vb.co - uvedge_b.va.co*rot
		#extract and transform island_b's boundary
		phantoms = {uvvertex: UVVertex(uvvertex.co*rot+trans, uvvertex.vertex) for uvvertex in other.verts}
		assert uvedge_b.va in phantoms and uvedge_b.vb in phantoms
		phantoms[uvedge_b.va] = uvedge_a.vb
		phantoms[uvedge_b.vb] = uvedge_a.va
		boundary_other = [UVEdge(phantoms[uvedge.va], phantoms[uvedge.vb], self) for uvedge in other.boundary_sorted if uvedge is not uvedge_b]
		#create event list
		sweepline = Sweepline()
		events_add = boundary_other + self.boundary_sorted
		events_add.remove(uvedge_a)
		events_remove = list(events_add)
		events_add.sort(reverse = True)
		events_remove.sort(key = lambda uvedge: uvedge.max, reverse = True)
		try:
			while events_remove:
				while events_add and events_add[-1].min <= events_remove[-1].max:
					sweepline.add(events_add.pop())
				sweepline.remove(events_remove.pop())
		except Intersection:
			print(":(", end=" ")
			return False
		except (RuntimeError, ValueError):
			s = 100
			print ("<path class='outer' d='")
			for edge in self.boundary_sorted:
				print("M {:.3f} {:.3f} L {:.3f} {:.3f}".format(edge.min.co.x*s, edge.min.co.y*s, edge.max.co.x*s, edge.max.co.y*s), end=" ")
			print("'/>")
			print ("<path d='")
			for edge in boundary_other:
				print("M {:.3f} {:.3f} L {:.3f} {:.3f}".format(edge.min.co.x*s, edge.min.co.y*s, edge.max.co.x*s, edge.max.co.y*s), end=" ")
			print("'/>")
			print ("<path d='")
			for edge in other.boundary_sorted:
				print("M {:.3f} {:.3f} L {:.3f} {:.3f}".format(edge.min.co.x*s, edge.min.co.y*s, edge.max.co.x*s, edge.max.co.y*s), end=" ")
			print("'/>")
			print ("Runtime error before {}".format(uvedge))
			raise
		
		print("Join!", end=" ")
		#remove edge from boundary
		self.boundary_sorted.remove(uvedge_a)
		other.boundary_sorted.remove(uvedge_b)
		edge.is_main_cut = False
		# apply transformation to the vertices

		# join other's data on self
		self.verts.update(phantoms.values())
		for uvedge in other.edges:
			uvedge.island = self
			uvedge.va = phantoms[uvedge.va]
			uvedge.vb = phantoms[uvedge.vb]
			uvedge.min, uvedge.max = (uvedge.va, uvedge.vb) if (uvedge.va < uvedge.vb) else (uvedge.vb, uvedge.va)
		self.edges.extend(other.edges)

		# DEBUG:
		for uvedge in other.boundary_sorted:
			assert uvedge in other.edges
		verts_from_edges = set()
		for uvedge in self.edges:
			verts_from_edges.add(uvedge.va)
			verts_from_edges.add(uvedge.vb)
		for va in verts_from_edges:
			for vb in verts_from_edges:
				if va.vertex.index == vb.vertex.index and (va.co-vb.co).length < 1e-5 and va is not vb:
					print ("-----------------------------------------------Not equal:", va, vb)
		if verts_from_edges != self.verts:
			print ("Verts:", self.verts)
			print ("Difference:", verts_from_edges - self.verts)
			print ("UVEdge A: {}, UVEdge B: {}".format(uvedge_a, uvedge_b))
			print ("Edges ({}):".format(len(self.edges)), self.edges)

		for uvface in other.faces:
			uvface.island = self
			uvface.verts = [phantoms[uvvertex] for uvvertex in uvface.verts]
			uvface.uvvertex_by_id = {index: phantoms[uvvertex] for index, uvvertex in uvface.uvvertex_by_id.items()}
		self.faces.extend(other.faces)
		self.boundary_sorted.extend(other.boundary_sorted)
		self.boundary_sorted.sort()
		
		#DEBUG
		sweepline = Sweepline()
		events_add = list(self.boundary_sorted)
		events_add.sort(reverse = True)
		events_remove = list(self.boundary_sorted)
		events_remove.sort(key = lambda uvedge: uvedge.max, reverse = True)
		try:
			while events_remove:
				while events_add and events_add[-1].min <= events_remove[-1].max:
					sweepline.add(events_add.pop())
				sweepline.remove(events_remove.pop())
				sweepline.dump()
		except (Intersection, RuntimeError):
			print("Problém, Houstone.")
			s = 100
			print ("<path class='outer' d='")
			for edge in self.boundary_sorted:
				print("M {:.3f} {:.3f} L {:.3f} {:.3f}".format(edge.min.co.x*s, edge.min.co.y*s, edge.max.co.x*s, edge.max.co.y*s), end=" ")
			print("'/>")
			raise RuntimeError()

		return True

	def add(self, uvface):
		self.verts.update(uvface.verts) #FIXME: why isn't uvface.verts a set directly?
		if uvface.is_sticker:
			self.stickers.append(uvface)
		else:
			self.faces.append(uvface)
	
	def generate_convex_hull(self) -> list:
		"""Returns a subset of self.verts that forms the best fitting convex polygon."""
		def make_convex_curve(verts):
			"""Remove vertices from given vert list so that the result poly is a convex curve (works for both top and bottom)."""
			i=1 #we can skip the first vertex as it is always convex
			while i<len(verts)-1:
				left_edge=verts[i].co-verts[i-1].co #edge from the current vertex to the left
				right_edge=verts[i+1].co-verts[i].co #edge -||- to the right
				# if slope to the left is not bigger than one to the right, the angle is concave
				if (left_edge.x==0 and right_edge.x==0) or \
						(not (left_edge.x==0 or right_edge.x==0) and #division by zero
						left_edge.y/left_edge.x <= right_edge.y/right_edge.x):
					verts.pop(i) #so let us omit this vertex
					i = max(1, i-1) #step back to check, but do not go below 1
				else:
					i += 1 #if the angle is convex, go ahead
			return verts
		self.verts=set()
		for face in self.faces + self.stickers:
			self.verts.update(face.verts)
		verts_top=list(self.verts)
		verts_top.sort(key=lambda vertex: vertex.co.x) #sorted left to right
		make_convex_curve(verts_top)
		verts_bottom=list(self.verts)
		verts_bottom.sort(key=lambda vertex: -vertex.co.x) #sorted right to left
		make_convex_curve(verts_bottom)
		#remove left and right ends and concatenate the lists to form a polygon in the right order
		verts_top.pop()
		verts_bottom.pop()
		return verts_top + verts_bottom
	
	def generate_bounding_box(self):
		"""Find the rotation for the optimal bounding box and calculate its dimensions."""
		def bounding_box_score(size):
			"""Calculate the score - the bigger result, the better box."""
			return 1/(size.x*size.y)
		verts_convex = self.generate_convex_hull()
		#DEBUG
		if len(verts_convex)==0:
			print ("papermodel ERROR: unable to calculate convex hull")
			return M.Vector((0,0))
		#go through all edges and search for the best solution
		best_box=(0, 0, M.Vector((0,0)), M.Vector((0,0))) #(score, angle, box) for the best score
		vertex_a = verts_convex[len(verts_convex)-1]
		for vertex_b in verts_convex:
			if vertex_b.co != vertex_a.co:
				angle=angle2d(vertex_b.co-vertex_a.co)
				rot=M.Matrix.Rotation(angle, 2)
				#find the dimensions in both directions
				bottom_left=M.Vector((0,0))
				top_right=M.Vector((0,0))
				verts_rotated=list(map(lambda vertex: vertex.co*rot, verts_convex))
				bottom_left.x=min(map(lambda vertex: vertex.x, verts_rotated))
				bottom_left.y=min(map(lambda vertex: vertex.y, verts_rotated))
				top_right.x=max(map(lambda vertex: vertex.x, verts_rotated))
				top_right.y=max(map(lambda vertex: vertex.y, verts_rotated))
				score = bounding_box_score(top_right-bottom_left)
				if score > best_box[0]:
					best_box = score, angle, top_right-bottom_left, bottom_left
			vertex_a=vertex_b
		#switch the box so that it is taller than wider
		if best_box[2].x>best_box[2].y:
			best_box=best_box[0], best_box[1]+pi/2, best_box[2].yx, M.Vector((-best_box[3].y-best_box[2].y, best_box[3].x))
		self.angle=best_box[1]
		self.bounding_box=best_box[2]
		self.offset=-best_box[3]
	
	def apply_scale(self, scale=1):
		if scale != 1:
			for vertex in self.verts:
				vertex.co *= scale
	
	def save_uv(self, tex, aspect_ratio=1):
		"""Save UV Coordinates of all UVFaces to a given UV texture
		tex: UV Texture layer to use (BPy MeshTextureFaceLayer struct)
		page_size: size of the page in pixels (vector)"""
		for uvface in self.faces:
			if not uvface.is_sticker:
				texface = tex.data[uvface.face.index]
				rot = M.Matrix.Rotation(self.angle, 2)
				for i, uvvertex in enumerate(uvface.verts):
					uv = uvvertex.co * rot + self.pos + self.offset
					texface.uv_raw[2*i] = uv.x / aspect_ratio
					texface.uv_raw[2*i+1] = uv.y

class Page:
	"""Container for several Islands"""
	def __init__(self, num=1):
		self.islands=list()
		self.image=None
		self.name="page"+str(num)
	def add(self, island):
		self.islands.append(island)

class UVVertex:
	"""Vertex in 2D"""
	def __init__(self, vector, vertex=None):
		if type(vector) is UVVertex: #Copy constructor
			self.co=vector.co.copy()
			self.vertex=vector.vertex
		else:
			self.co=(M.Vector(vector)).xy
			self.vertex=vertex
	def __hash__(self):
		if self.vertex:
			return self.vertex.index
		else:
			return int(hash(self.co.x)+hash(self.co.y))
	def __sub__(self, other):
		return (self.co - other.co).to_3d()
	def __str__(self):
		if self.vertex:
			return "UV "+str(self.vertex.index)+" ["+strf(self.co.x)+", "+strf(self.co.y)+"]"
		else:
			return "UV * ["+strf(self.co.x)+", "+strf(self.co.y)+"]"
	def __repr__(self):
		return str(self)
	def __eq__(self, other):
		return (self is other) or (self.vertex == other.vertex and self.co.x == other.co.x and self.co.y == other.co.y)
	def __ne__(self, other):
		return not self == other
	def __lt__(self, other):
		return self.co.x < other.co.x or (self.co.x == other.co.x and self.co.y < other.co.y)
	def __le__(self, other):
		return (self is other) or (self.co.x < other.co.x) or (self.co.x == other.co.x and self.co.y <= other.co.y)
	def __ge__(self, other):
		return (self is other) or (self.co.x > other.co.x) or (self.co.x == other.co.x and self.co.y >= other.co.y)

class UVEdge:
	"""Edge in 2D"""
	def __init__(self, vertex1:UVVertex, vertex2:UVVertex, island:Island, uvface=None, edge:Edge=None):
		self.va = vertex1
		self.vb = vertex2
		self.min, self.max = (vertex1, vertex2) if (vertex1 < vertex2) else (vertex2, vertex1)
		self.island = island
		if edge:
			self.edge = edge
			edge.uvedges.append(self)
		#Every UVEdge is attached to only one UVFace. UVEdges are doubled as needed, because they both have to point clockwise around their faces
		self.uvface = uvface
	def __lt__(self, other):
		return self.min < other.min
	def __le__(self, other):
		return self.min <= other.min
	def __str__(self):
		#return "({} - {})".format(self.min, self.max)
		return "({} - {})".format(self.va, self.vb)
	def __repr__(self):
		return str(self)
		

class UVFace:
	"""Face in 2D"""
	is_sticker=False
	def __init__(self, face:Face, island:Island):
		"""Creace an UVFace from a Face and a fixed edge.
		face: Face to take coordinates from
		island: Island to register itself in
		fixed_edge: Edge to connect to (that already has UV coordinates)"""
		if type(face) is Face:
			self.verts=list()
			self.face=face
			face.uvface=self
			self.island=island
			rot=z_up_matrix(face.normal)
			self.uvvertex_by_id=dict() #link vertex id -> UVVertex
			for vertex in face.verts:
				uvvertex=UVVertex(vertex.co*rot, vertex)
				self.verts.append(uvvertex)
				self.uvvertex_by_id[vertex.index]=uvvertex
			#DEBUG: check lengths
			diff=1
			for (va, uva), (vb, uvb) in pairs(zip(face.verts, self.verts)):
				diff *= (va.co-vb.co).length/(uva.co-uvb.co).length
			global differences
			differences.append((diff, face.normal))
		elif type(face) is UVFace: #copy constructor TODO: DOES NOT WORK
			self.verts=list(face.verts)
			self.face=face.face
			self.uvvertex_by_id=dict(face.uvvertex_by_id)
		self.edges=list()
		edge_by_verts=dict()
		for edge in face.edges:
			edge_by_verts[(edge.va.index, edge.vb.index)]=edge
			edge_by_verts[(edge.vb.index, edge.va.index)]=edge
		for va, vb in pairs(self.verts):
			uvedge = UVEdge(va, vb, island, self, edge_by_verts[(va.vertex.index, vb.vertex.index)])
			self.edges.append(uvedge)
			island.edges.append(uvedge)
		#self.edges=[UVEdge(self.uvvertex_by_id[edge.va.data.index], self.uvvertex_by_id[edge.vb.data.index], island, edge) for edge in face.edges]
		#DEBUG:
		self.check("construct")
	
	def __lt__(self, other):
		"""Hack for usage in heaps"""
		return self.face.index < other.face.index
	
	def __repr__(self):
		return "UVFace("+str(self.face.index)+")"
	
	#DEBUG:
	def check(self, message=""):
		for this_uvedge in self.edges:
			if this_uvedge.va not in self.verts:
				print("papermodel ERROR: My UVEdge doesn't belong to myself",this_uvedge.va.vertex.index, object.__repr__(this_uvedge), message)
			if this_uvedge.vb not in self.verts:
				print("papermodel ERROR: My UVEdge doesn't belong to myself",this_uvedge.vb.vertex.index, object.__repr__(this_uvedge), message)
		for vertex_id, vertex in self.uvvertex_by_id.items():
			if vertex not in self.verts:
				print("papermodel ERROR: UVVertex found by ID does not exist")
		if len(self.verts) > 3:
			if (self.verts[0].co-self.verts[2].co).angle(self.verts[1].co-self.verts[3].co)<pi/10:
				print("papermodel NOTICE: This face is weirdly twisted",self.face.index, message)
		
class Sticker(UVFace):
	"""Sticker face"""
	is_sticker=True
	def __init__(self, uvedge, default_width=0.005, faces=list(), other_face=None):
		"""Sticker is directly appended to the edge given by two UVVerts"""
		#faces is a placeholder: it should contain all possibly overlaping faces
		#other_face is a placeholder too: that should be the sticking target and this sticker must fit into it
		edge=uvedge.va.co-uvedge.vb.co
		sticker_width=min(default_width, edge.length/2)
		other=uvedge.edge.other_uvedge(uvedge) #This is the other uvedge - the sticking target
		other_edge=other.vb.co-other.va.co
		cos_a=cos_b=0.5 #angle a is at vertex uvedge.va, b is at uvedge.vb
		sin_a=sin_b=0.75**0.5
		len_a=len_b=sticker_width/sin_a #len_a is length of the side adjacent to vertex a, len_b similarly
		#fix overlaps with the most often neighbour - its sticking target
		if uvedge.va==other.vb:
			cos_a=min(max(cos_a, (edge*other_edge)/(edge.length**2)), 1) #angles between pi/3 and 0; fix for math errors
			sin_a=(1-cos_a**2)**0.5
			len_b=min(len_a, (edge.length*sin_a)/(sin_a*cos_b+sin_b*cos_a))
			if sin_a==0:
				len_a=0
			else:
				len_a=min(sticker_width/sin_a, (edge.length-len_b*cos_b)/cos_a)
		elif uvedge.vb==other.va:
			cos_b=min(max(cos_b, (edge*other_edge)/(edge.length**2)), 1) #angles between pi/3 and 0; fix for math errors
			sin_b=(1-cos_b**2)**0.5
			len_a=min(len_a, (edge.length*sin_b)/(sin_a*cos_b+sin_b*cos_a))
			if sin_b==0:
				len_b=0
			else:
				len_b=min(sticker_width/sin_b, (edge.length-len_a*cos_a)/cos_b)
		v3 = uvedge.vb.co + edge * M.Matrix(((cos_b, sin_b), (-sin_b, cos_b)))*len_b/edge.length
		v4 = uvedge.va.co + edge * M.Matrix(((-cos_a, sin_a), (-sin_a, -cos_a)))*len_a/edge.length
		if v3!=v4:
			self.verts=[uvedge.vb, UVVertex(v3), UVVertex(v4), uvedge.va]
		else:
			self.verts=[uvedge.vb, UVVertex(v3), uvedge.va]
		"""for face in faces: #TODO: fix all overlaps
			self.cut(face) #yep, this is slow
		if other_face: 
			dupli=UVFace(other_face, edge=edge)"""
	def cut(self, face):
		"""Cut the given UVFace's area from this sticker (if they overlap) - placeholder."""
		#TODO
		pass

class SVG:
	"""Simple SVG exporter"""
	def __init__(self, page_size_pixels:M.Vector, pure_net=True):
		"""Initialize document settings.
		page_size_pixels: document dimensions in pixels
		pure_net: if True, do not use image"""
		self.page_size = page_size_pixels
		self.scale = page_size_pixels.y
		self.pure_net = pure_net
	def add_mesh(self, mesh):
		"""Set the Mesh to process."""
		self.mesh=mesh
	def format_vertex(self, vector, rot=1, pos=M.Vector((0,0))):
		"""Return a string with both coordinates of the given vertex."""
		vector = vector*rot + pos
		return str(vector.x*self.scale) + " " + str((1-vector.y)*self.scale)
	def write(self, filename):
		"""Write data to a file given by its name."""
		line_through = " L ".join #utility function
		for num, page in enumerate(self.mesh.pages):
			with open(filename+"_"+page.name+".svg", 'w') as f:
				f.write("<?xml version='1.0' encoding='UTF-8' standalone='no'?>")
				f.write("<svg xmlns:svg='http://www.w3.org/2000/svg' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' version='1.1' width='" + str(self.page_size.x) + "px' height='" + str(self.page_size.y) + "px'>")
				f.write("""<style type="text/css">
					path {fill:none; stroke-width:1px; stroke-linecap:square; stroke-linejoin:bevel; stroke-dasharray:none}
					path.concave {stroke:#000; stroke-dasharray:8,4,2,4; stroke-dashoffset:0}
					path.convex {stroke:#000; stroke-dasharray:4,8; stroke-dashoffset:0}
					path.outer {stroke:#000; stroke-dasharray:none; stroke-width:1.5px}
					path.background {stroke:#fff}
					path.outer_background {stroke:#fff; stroke-width:2px}
					path.sticker {fill: #fff; stroke: #000; fill-opacity: 0.4; stroke-opacity: 0.7}
					rect {fill:#ccc; stroke:none}
				</style>""")
				if not self.pure_net:
					f.write("<image x='0' y='0' width='" + str(self.page_size.x) + "' height='" + str(self.page_size.y) + "' xlink:href='file://" + filename + "_" + page.name + ".png'/>")
				f.write("<g>")
				for island in page.islands:
					f.write("<g>")
					rot = M.Matrix.Rotation(island.angle, 2)
					#FIXME: join a list of strings instead. This sucks.
					data_outer = data_convex = data_concave = data_stickers = ""
					for uvedge in island.edges:
						data_uvedge = "\nM " + line_through([self.format_vertex(vertex.co, rot, island.pos + island.offset) for vertex in [uvedge.va, uvedge.vb]])
						#FIXME: The following clause won't return correct results for uncut edges with more than two faces connected
						if uvedge.edge.is_cut(uvedge.uvface.face):
							assert uvedge in uvedge.uvface.island.boundary_sorted
							data_outer += data_uvedge
						else:
							if uvedge.va.vertex.index > uvedge.vb.vertex.index: #each edge is in two opposite-oriented variants; we want to add each only once
								angle = uvedge.edge.angles[uvedge.uvface.face]
								if angle > 0.01:
									data_convex += data_uvedge
								elif angle < -0.01:
									data_concave += data_uvedge
					if island.stickers:
						f.write("<g>")
						for sticker in island.stickers: #Stickers are separate paths in one group
							f.write("<path class='sticker' d='M " + line_through([self.format_vertex(vertex.co, rot, island.pos + island.offset) for vertex in sticker.verts]) + " Z'/>")
						f.write("</g>")
					if data_outer: 
						if not self.pure_net:
							f.write("<path class='outer_background' d='" + data_outer + "'/>")
						f.write("<path class='outer' d='" + data_outer + "'/>")
					if not self.pure_net and (data_convex or data_concave):
						f.write("<path class='background' d='" + data_convex + data_concave + "'/>")
					if data_convex: f.write("<path class='convex' d='" + data_convex+"'/>")
					if data_concave: f.write("<path class='concave' d='" + data_concave+"'/>")
					f.write("</g>")
				f.write("</g>")
				f.write("</svg>")
				f.close()

class MakeUnfoldable(bpy.types.Operator):
	"""Blender Operator: unfold the selected object."""
	bl_idname = "mesh.make_unfoldable"
	bl_label = "Make Unfoldable"
	bl_description = "Mark seams so that the mesh can be exported as a paper model"
	bl_options = {'REGISTER', 'UNDO'}
	edit = bpy.props.BoolProperty(name="", description="", default=False, options={'HIDDEN'})
	priority_effect_convex = bpy.props.FloatProperty(name="Priority Convex", description="Priority effect for edges in convex angles", default=priority_effect["convex"], soft_min=-1, soft_max=10, subtype='FACTOR')
	priority_effect_concave = bpy.props.FloatProperty(name="Priority Concave", description="Priority effect for edges in concave angles", default=priority_effect["concave"], soft_min=-1, soft_max=10, subtype='FACTOR')
	priority_effect_length = bpy.props.FloatProperty(name="Priority Length", description="Priority effect of edge length", default=priority_effect["length"], soft_min=-10, soft_max=1, subtype='FACTOR')
	
	@classmethod
	def poll(cls, context):
		return context.active_object and context.active_object.type=="MESH"
		
	def draw(self, context):
		layout = self.layout
		layout.label(text="Edge Cutting Factors:")
		col = layout.column(align=True)
		col.label(text="Face Angle:")
		col.prop(self.properties, "priority_effect_convex", text="Convex")
		col.prop(self.properties, "priority_effect_concave", text="Concave")
		layout.prop(self.properties, "priority_effect_length", text="Edge Length")
	
	def execute(self, context):
		global priority_effect
		props = self.properties
		sce = bpy.context.scene
		priority_effect['convex']=props.priority_effect_convex
		priority_effect['concave']=props.priority_effect_concave
		priority_effect['length']=props.priority_effect_length
		orig_mode=context.object.mode
		bpy.ops.object.mode_set(mode='OBJECT')
		display_islands = sce.io_paper_model_display_islands
		sce.io_paper_model_display_islands = False

		unfolder = Unfolder(context.active_object)
		unfolder.prepare()

		island_list = context.scene.island_list
		while island_list:
			#remove previously defined islands
			island_list.remove(0)
		for island in unfolder.mesh.islands:
			#add islands to UI list and set default descriptions
			list_item = island_list.add()
			list_item.name = "Island ({} faces)".format(len(island.faces))
			#add faces' IDs to the island
			for uvface in island.faces:
				face_list_item = list_item.faces.add()
				face_list_item.id = uvface.face.index
		sce.island_list_index = -1
		list_selection_changed(sce, bpy.context)

		unfolder.mesh.data.show_edge_seams=True
		bpy.ops.object.mode_set(mode=orig_mode)
		sce.io_paper_model_display_islands = display_islands
		#global twisted_quads
		#if len(twisted_quads) > 0:
		#	self.report(type="ERROR_INVALID_INPUT", message="There are twisted quads in the model, you should divide them to triangles. Use the 'Twisted Quads' option in View Properties panel to see them.")
		return {'FINISHED'}

class ExportPaperModel(bpy.types.Operator):
	"""Blender Operator: save the selected object's net and optionally bake its texture"""
	bl_idname = "export_mesh.paper_model"
	bl_label = "Export Paper Model"
	bl_description = "Export the selected object's net and optionally bake its texture"
	filepath = bpy.props.StringProperty(name="File Path", description="Target file to save the SVG")
	filename = bpy.props.StringProperty(name="File Name", description="Name of the file")
	directory = bpy.props.StringProperty(name="Directory", description="Directory of the file")
	output_size_x = bpy.props.FloatProperty(name="Page Size X", description="Page width", default=0.210, soft_min=0.105, soft_max=0.841, subtype="UNSIGNED", unit="LENGTH")
	output_size_y = bpy.props.FloatProperty(name="Page Size Y", description="Page height", default=0.297, soft_min=0.148, soft_max=1.189, subtype="UNSIGNED", unit="LENGTH")
	output_dpi = bpy.props.FloatProperty(name="Unfolder DPI", description="Output resolution in points per inch", default=90, min=1, soft_min=30, soft_max=600, subtype="UNSIGNED")
	output_pure = bpy.props.BoolProperty(name="Pure Net", description="Do not bake the bitmap", default=True)
	bake_selected_to_active = bpy.props.BoolProperty(name="Selected to Active", description="Bake selected to active (if not exporting pure net)", default=True)
	sticker_width = bpy.props.FloatProperty(name="Tab Size", description="Width of gluing tabs", default=0.005, soft_min=0, soft_max=0.05, subtype="UNSIGNED", unit="LENGTH")
	model_scale = bpy.props.FloatProperty(name="Scale", description="Coefficient of all dimensions when exporting", default=1, soft_min=0.0001, soft_max=1.0, subtype="FACTOR")
	unfolder=None
	largest_island_ratio=0
	
	@classmethod
	def poll(cls, context):
		return context.active_object and context.active_object.type=="MESH"
	
	def execute(self, context):
		try:
			self.unfolder.save(self.properties)
			return {'FINISHED'}
		except UnfoldError as error:
			self.report(type={'ERROR_INVALID_INPUT'}, message=error.args[0])
			return {'CANCELLED'}
		except:
			raise
	def get_scale_ratio(self):
		return self.unfolder.mesh.largest_island_ratio(M.Vector((self.properties.output_size_x, self.properties.output_size_y))) * self.properties.model_scale
	def invoke(self, context, event):
		sce=context.scene
		self.properties.bake_selected_to_active = sce.render.use_bake_selected_to_active
		
		self.unfolder=Unfolder(context.active_object)
		self.unfolder.prepare(self.properties)
		scale_ratio = self.get_scale_ratio()
		if scale_ratio > 1:
			self.properties.model_scale = 0.95/scale_ratio
		wm = context.window_manager
		wm.fileselect_add(self)
		return {'RUNNING_MODAL'}
	
	def draw(self, context):
		layout = self.layout
		col = layout.column(align=True)
		col.label(text="Page size:")
		col.prop(self.properties, "output_size_x")
		col.prop(self.properties, "output_size_y")
		layout.prop(self.properties, "output_dpi")
		layout.label(text="Model scale:")
		layout.prop(self.properties, "model_scale")
		scale_ratio = self.get_scale_ratio()
		if scale_ratio > 1:
			layout.label(text="An island is "+strf(scale_ratio)+"x bigger than page", icon="ERROR")
		elif scale_ratio > 0:
			layout.label(text="Largest island is 1/"+strf(1/scale_ratio)+" of page")
		layout.prop(self.properties, "output_pure")
		col = layout.column()
		col.active = not self.properties.output_pure
		col.prop(self.properties, "bake_selected_to_active")
		layout.label(text="Document settings:")
		layout.prop(self.properties, "sticker_width")

""" 
class VIEW3D_paper_model(bpy.types.Panel):
	Blender UI Panel definition for Unfolder
	bl_space_type = 'VIEW_3D'
	bl_region_type = 'TOOLS'
	bl_label = "Export Paper Model"
	
	bpy.types.Scene.unfolder_output_size_x = bpy.props.FloatProperty(name="Page Size X", description="Page width", default=0.210, soft_min=0.105, soft_max=0.841, subtype="UNSIGNED", unit="LENGTH")
	bpy.types.Scene.unfolder_output_size_y = bpy.props.FloatProperty(name="Page Size Y", description="Page height", default=0.297, soft_min=0.148, soft_max=1.189, subtype="UNSIGNED", unit="LENGTH")
	bpy.types.Scene.unfolder_output_dpi = bpy.props.FloatProperty(name="Unfolder DPI", description="Output resolution in points per inch", default=90, min=1, soft_min=30, soft_max=600, subtype="UNSIGNED")
	bpy.types.Scene.unfolder_output_pure = bpy.props.BoolProperty(name="Pure Net", description="Do not bake the bitmap", default=True)
	
	@classmethod
	def poll(cls, context):
		return (context.active_object and context.active_object.type == 'MESH')

	def draw(self, context):
		layout = self.layout
		layout.operator("mesh.make_unfoldable")
		col = layout.column()
		sub = col.column(align=True)
		sub.label(text="Page size:")
		sub.prop(bpy.context.scene, "unfolder_output_size_x", text="Width")
		sub.prop(bpy.context.scene, "unfolder_output_size_y", text="Height")
		col.prop(bpy.context.scene, "unfolder_output_dpi", text="DPI")
		col.prop(bpy.context.scene, "unfolder_output_pure")
		sub = col.column()
		sub.active = not context.scene.unfolder_output_pure
		sub.prop(context.scene.render, "use_bake_selected_to_active", text="Bake Selected to Active")
		col.operator("export.paper_model", text="Export Net...")
"""

def menu_func(self, context):
	self.layout.operator("export_mesh.paper_model", text="Paper Model (.svg)")

class VIEW3D_PT_paper_model(bpy.types.Panel):
	bl_label = "Paper Model"
	bl_space_type = "VIEW_3D"
	bl_region_type = "TOOLS"

	def draw(self, context):
		layout = self.layout
		sce = context.scene
		layout.operator("mesh.make_unfoldable")
		box = layout.box()
		box.label(text="{} island(s):".format(len(sce.island_list)))
		box.template_list(sce, 'island_list', sce, 'island_list_index', rows=1, maxrows=5)
		layout.prop(sce, "io_paper_model_display_labels", icon='RESTRICT_VIEW_OFF')
		box.prop(sce, "io_paper_model_display_islands")#, icon='RESTRICT_VIEW_OFF')
		sub = box.row()
		sub.active = sce.io_paper_model_display_islands
		sub.prop(sce, "io_paper_model_islands_alpha")
		
		island_list = sce.island_list
		if sce.island_list_index >= 0 and len(island_list) > 0:
			list_item = island_list[sce.island_list_index]
			box.prop(list_item, "label")
		layout.operator("export_mesh.paper_model")
	
def display_islands(self, context):
	import bgl
	#TODO: save the vertex positions and don't recalculate them always
	#TODO: don't use active object, but rather save the object itself
	perspMatrix = context.space_data.region_3d.perspective_matrix
	perspBuff = bgl.Buffer(bgl.GL_FLOAT, 16, [perspMatrix[i][j] for i in range(4) for j in range(4)])
	bgl.glLoadIdentity()
	bgl.glMatrixMode(bgl.GL_PROJECTION)
	bgl.glLoadMatrixf(perspBuff)
	bgl.glEnable(bgl.GL_BLEND)
	bgl.glBlendFunc (bgl.GL_SRC_ALPHA, bgl.GL_ONE_MINUS_SRC_ALPHA);
	bgl.glEnable(bgl.GL_POLYGON_OFFSET_FILL)
	bgl.glPolygonOffset(0.0, -2.0) #offset in Zbuffer to remove flicker
	bgl.glPolygonMode(bgl.GL_FRONT_AND_BACK, bgl.GL_FILL)
	bgl.glColor4f(1.0, 0.4, 0.0, self.io_paper_model_islands_alpha)
	ob = bpy.context.active_object
	mesh = ob.data
	global highlight_faces
	for face_id in highlight_faces:
		face = mesh.faces[face_id]
		bgl.glBegin(bgl.GL_POLYGON)
		for vertex_id in face.vertices:
			vertex = mesh.vertices[vertex_id]
			co = vertex.co.copy()
			co.resize_4d()
			co = co * ob.matrix_world
			co /= co[3]
			bgl.glVertex3f(co[0], co[1], co[2])
		bgl.glEnd()
	bgl.glPolygonOffset(0.0, 0.0)
	bgl.glDisable(bgl.GL_POLYGON_OFFSET_FILL)
display_islands.handle = None

def display_labels(self, context):
	import bgl, blf, mathutils
	view_mat = context.space_data.region_3d.perspective_matrix
	
	global labels
	mid_x = context.region.width/2.0
	mid_y = context.region.height/2.0
	width = context.region.width
	height = context.region.height
	bgl.glColor3f(1,1,0)
	for position, label in labels.values():
		position.resize_4d()
		vec = position * view_mat
		vec /= vec[3]
		x = int(mid_x + vec[0]*width/2.0)
		y = int(mid_y + vec[1]*height/2.0)
		blf.position(0, x, y, 0)
		blf.draw(0, label)
display_labels.handle = None

def display_labels_changed(self, context):
	"""Switch displaying labels on/off"""
	region = [region for region in context.area.regions if region.type=='WINDOW'][0]
	if self.io_paper_model_display_labels:
		if not display_labels.handle:
			display_labels.handle = region.callback_add(display_labels, (self, context), "POST_PIXEL")
	else:
		if display_labels.handle:
			region.callback_remove(display_labels.handle)
			display_labels.handle = None

def display_islands_changed(self, context):
	"""Switch highlighting islands on/off"""
	region = [region for region in context.area.regions if region.type=='WINDOW'][0]
	if self.io_paper_model_display_islands:
		if not display_islands.handle:
			display_islands.handle = region.callback_add(display_islands, (self, context), "POST_VIEW")
	else:
		if display_islands.handle:
			region.callback_remove(display_islands.handle)
			display_islands.handle = None

def list_selection_changed(self, context):
	"""Update the island highlighted in 3D View"""
	global highlight_faces
	if self.island_list_index >= 0:
		list_item = self.island_list[self.island_list_index]
		highlight_faces = [face.id for face in list_item.faces]
	else:
		highlight_faces = list()
	"""
	mesh = bpy.context.active_object.data
	face_data = list()
	for vertex_id in mesh.faces[face_id].vertices:
		face_data.append(mesh.vertices[vertex_id].co)
	highlight_faces.append(face_data)
	"""
def label_changed(self, context):
	self.name = "{} ({} faces)".format(self.label, len(self.faces))

class FaceList(bpy.types.PropertyGroup):
	id = bpy.props.IntProperty(name="Face ID")
class IslandList(bpy.types.PropertyGroup):
	faces = bpy.props.CollectionProperty(type=FaceList, name="Faces", description="Faces belonging to this island")
	label = bpy.props.StringProperty(name="Label", description="*out of order* Label on this island", default="No Label", update=label_changed)
bpy.utils.register_class(FaceList)
bpy.utils.register_class(IslandList)

def register():
	bpy.utils.register_module(__name__)

	bpy.types.Scene.io_paper_model_display_labels = bpy.props.BoolProperty(name="Display edge priority", description="*debug property*", update=display_labels_changed)
	bpy.types.Scene.io_paper_model_display_islands = bpy.props.BoolProperty(name="Highlight selected island", update=display_islands_changed)
	bpy.types.Scene.io_paper_model_islands_alpha = bpy.props.FloatProperty(name="Highlight Alpha", description="Alpha value for island highlighting", min=0.0, max=1.0, default=0.3)
	bpy.types.Scene.io_paper_model_display_quads = bpy.props.BoolProperty(name="Highlight tilted quads", description="*out of order* Highlight tilted quad faces that would be distorted by export")
	bpy.types.Scene.island_list = bpy.props.CollectionProperty(type=IslandList, name= "Island List", description= "")
	bpy.types.Scene.island_list_index = bpy.props.IntProperty(name="Island List Index", default= -1, min= -1, max= 100, update=list_selection_changed)
	bpy.types.INFO_MT_file_export.append(menu_func)

def unregister():
	bpy.utils.unregister_module(__name__)
	bpy.types.INFO_MT_file_export.remove(menu_func)

if __name__ == "__main__":
	register()
