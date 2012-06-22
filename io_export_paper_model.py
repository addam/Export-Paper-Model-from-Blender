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

#### FIXME:
# check that edges with 0 or 1 faces need not be marked as cut

#### TODO:
# choose edge's main pair of faces intelligently
# split islands bigger than selected page size
# UI elements to set line thickness and page size conveniently

bl_info = {
	"name": "Export Paper Model",
	"author": "Addam Dominec",
	"version": (0, 8),
	"blender": (2, 6, 3),
	"api": 48011,
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
try:
	from blist import blist
except ImportError:
	blist = list

priority_effect={
	'convex':0.5,
	'concave':1,
	'length':-0.05}
lines = list() #TODO: currently lines are not used
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
	if not vectlist:
		return M.Vector((0,0))
	last = vectlist[0]
	if type(last) is Vertex:
		vect_sum = last.co.copy() #keep the dimensions
		vect_sum.zero()
		for vect in vectlist:
			vect_sum += vect.co
	else:
		vect_sum = last.copy() #keep the dimensions
		vect_sum.zero()
		for vect in vectlist:
			vect_sum += vect
	return vect_sum / len(vectlist)

def angle2d(direction, unit_vector = M.Vector((1,0))):
	"""Get the view angle of point from origin."""
	if direction.length_squared == 0:
		raise ValueError("Zero vector does not define an angle")
	if len(direction) >= 3: #for 3d vectors
		direction = direction.to_2d()
	angle = unit_vector.angle_signed(direction)
	return angle

def cross_product(v1, v2):
	return v1.x*v2.y - v1.y*v2.x

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
		(+v1.x*v2.x +v1.y*v2.y, +v1.y*v2.x -v1.x*v2.y),
		(+v1.x*v2.y -v1.y*v2.x, +v1.x*v2.x +v1.y*v2.y)))

def z_up_matrix(n):
	"""Get a rotation matrix that aligns given vector upwards."""
	b=n.xy.length
	l=n.length
	if b>0:
		return M.Matrix((
			(n.x*n.z/(b*l),	n.y*n.z/(b*l), -b/l),
			(       -n.y/b,         n.x/b,    0),
			(            0,             0,    0)))
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

	def prepare(self, properties=None, mark_seams=False):
		"""Something that should be part of the constructor - TODO """
		self.mesh.generate_cuts()
		self.mesh.finalize_islands()
		self.mesh.save_uv()
		if mark_seams:
			self.mesh.mark_cuts()

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
			tex = self.mesh.save_uv(aspect_ratio = page_size.x / page_size.y)
			if not tex:
				raise UnfoldError("The mesh has no UV Map slots left. Either delete an UV Map or export pure net only.")
			#TODO: do we really need a switch of our own?
			selected_to_active = bpy.context.scene.render.use_bake_selected_to_active; bpy.context.scene.render.use_bake_selected_to_active = properties.bake_selected_to_active
			self.mesh.save_image(tex, filepath, page_size * ppm)
			#revoke settings
			bpy.context.scene.render.use_bake_selected_to_active=selected_to_active
		svg=SVG(page_size * ppm, properties.output_pure, properties.line_thickness)
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
		for bpy_face in mesh.polygons:
			face = Face(bpy_face, self)
			self.faces[bpy_face.index] = face
		for index in self.edges:
			self.edges[index].calculate_angle()
	
	def generate_cuts(self):
		"""Cut the mesh so that it will be unfoldable."""
		global twisted_quads, labels
		twisted_quads = list()
		labels = dict()
		#Silently check that all quads are flat
		global differences
		differences = list()
		
		for index in self.faces:
			self.faces[index].check_twisted()
		
		self.islands = {Island(face) for face in self.faces.values()}
		# check for edges that are cut permanently
		edges = [edge for edge in self.edges.values() if not edge.force_cut and len(edge.faces) > 1]
		if not edges:
			return True
		
		average_length = sum(edge.length for edge in edges) / len(edges)
		for edge in edges:
			edge.generate_priority(average_length)
		edges.sort(key = lambda edge:edge.priority, reverse=False)
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
			print ("""Papermodel warning: there are non-flat faces, which will be deformed in the output image. You should consider converting the mesh to triangle faces.
			Showing first five values (normally they should be very close to 1.0):""")
			for diff in differences[0:5]:
				print ("{:.5f}".format(diff[0]))
		return True
	
	def mark_cuts(self):
		"""Mark cut edges in the original mesh so that the user can see"""
		for edge in self.edges.values():
			edge.data.use_seam = edge.is_main_cut
	
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
				island.pos = M.Vector((self.boundary.x, self.boundary.y))
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
		#TODO: at first, it should cut all islands that are too big to fit the page
		#TODO: there should be a list of points off the boundary that are created from pairs of open edges
		largest_island_ratio = self.largest_island_ratio(page_size) 
		if largest_island_ratio > 1:
			raise UnfoldError("An island is too big to fit to the page size. To make the export possible, scale the object down "+strf(largest_island_ratio)+" times.")
		islands=list(self.islands)
		#sort islands by their ugliness (we need an ugly expression to treat ugliness correctly)
		islands.sort(reverse = True, key = lambda island: island.bounding_box.x**2 + island.bounding_box.y**2 + (island.bounding_box.x-island.bounding_box.y)**2)
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
						if island.bounding_box.x <= point.area.x and island.bounding_box.y <= point.area.y:
							point.add_island(island)
							island.is_placed = True
							page.add(island)
							remaining_count -= 1
							break
					points.sort(key=lambda point: point.niceness) #ugly points first (to get rid of them)
			self.pages.append(page)
	
	def save_uv(self, aspect_ratio=1): #page_size is in pixels
		bpy.ops.object.mode_set()
		#note: expecting that the active object's data is self.mesh
		tex = self.data.uv_textures.new()
		if not tex:
			return None
		tex.name = "Unfolded"
		tex.active = True
		loop = self.data.uv_layers[self.data.uv_layers.active_index]
		for island in self.islands:
			island.save_uv(loop, aspect_ratio)
		return tex
	
	def save_image(self, tex, filename, page_size_pixels:M.Vector):
		rd=bpy.context.scene.render
		recall_margin=rd.bake_margin; rd.bake_margin=0
		recall_clear=rd.use_bake_clear; rd.use_bake_clear=False

		tex.active = True
		loop = self.data.uv_layers[self.data.uv_layers.active_index]
		aspect_ratio = page_size_pixels.x / page_size_pixels.y
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
			texfaces=tex.data
			for island in page.islands:
				for uvface in island.faces:
					if not uvface.is_sticker:
						texfaces[uvface.face.index].image=image
				#island.save_uv(loop, aspect_ratio)
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
		self.uvedges=list()
		
		self.force_cut = bool(edge.use_seam) # such edges will always be cut
		self.is_main_cut = True # defines whether the first two faces are connected; all the others will be automatically treated as cut
		self.priority=None
		self.angle = None
		self.va.edges.append(self)
		self.vb.edges.append(self)
	
	def calculate_angle(self):
		"""Choose two main faces and calculate the angle between them"""
		if len(self.faces)==0:
			return
		elif len(self.faces)==1:
			self.angle = pi
			return
		else:
			face_a, face_b = self.faces[:2]
			# correction if normals are flipped
			a_is_clockwise = ((face_a.verts.index(self.vb) - face_a.verts.index(self.va)) % len(face_a.verts) == 1)
			b_is_clockwise = ((face_b.verts.index(self.va) - face_b.verts.index(self.vb)) % len(face_b.verts) == 1)
			if face_a.uvface and face_b.uvface:
				a_is_clockwise ^= face_a.uvface.flipped
				b_is_clockwise ^= face_b.uvface.flipped
			if a_is_clockwise == b_is_clockwise:
				if a_is_clockwise == (face_a.normal.cross(face_b.normal).dot(self.vect) > 0):
					self.angle = face_a.normal.angle(face_b.normal) # the angle is convex
				else:
					self.angle = -face_a.normal.angle(face_b.normal) # the angle is concave
			else:
				self.angle = face_a.normal.angle(-face_b.normal) # normals are flipped, so we know nothing
				# but let us be optimistic and treat the angle as convex :)

	def generate_priority(self, average_length=1):
		"""Calculate initial priority value."""
		angle = self.angle
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
		if face is None or self.faces.index(face) < 2:
			return self.is_main_cut
		#All other faces (third and more) are automatically treated as cut
		else:
			return True
	
	def __str__(self):
		return "Edge id: {}".format(self.data.index)
	def __repr__(self):
		return "Edge(id={}...)".format(self.data.index)
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
		self.loop_start = bpy_face.loop_start
		self.uvface = None
		
		#TODO: would be nice to reuse the existing normal if possible
		if len(self.verts) == 3:
			# normal of a triangle can be calculated directly
			self.normal = (self.verts[1]-self.verts[0]).cross(self.verts[2]-self.verts[0]).normalized()
		else:
			# Newell's method
			nor = M.Vector((0,0,0))
			for a, b in pairs(self.verts):
				p, m = a+b, a-b
				nor.x, nor.y, nor.z = nor.x+m.y*p.z, nor.y+m.z*p.x, nor.z+m.x*p.y
			self.normal = nor.normalized()
		for verts_indices in bpy_face.edge_keys:
			edge = mesh.edges_by_verts_indices[verts_indices]
			self.edges.append(edge)
			edge.faces.append(self)
	def check_twisted(self):
		if len(self.verts) > 3:
			global twisted_quads, lines
			center = vectavg(self.verts)
			plane_d = center.dot(self.normal)
			diameter = max((center-vertex.co).length for vertex in self.verts)
			for vertex in self.verts:
				# check coplanarity
				if abs(vertex.co.dot(self.normal) - plane_d) > diameter*0.01: #TODO: this threshold should be editable or well chosen at least
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
		self.edges=set()
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
			
		def is_below(self: UVEdge, other: UVEdge, cross=cross_product):
			#TODO? estimate the epsilon based on input vectors
			if self is other:
				return False
			if self.top < other.bottom:
				return True
			if other.top < self.bottom:
				return False
			if self.max.tup <= other.min.tup:
				return True
			if other.max.tup <= self.min.tup:
				return False
			self_vector = self.max - self.min
			min_to_min = other.min - self.min
			cross_b1 = cross(self_vector, min_to_min)
			cross_b2 = cross(self_vector, (other.max - self.min))
			if cross_b1 != 0 or cross_b2 != 0:
				if cross_b1 >= 0 and cross_b2 >= 0:
					return True
				if cross_b1 <= 0 and cross_b2 <= 0:
					return False
			other_vector = other.max - other.min
			cross_a1 = cross(other_vector, -min_to_min)
			cross_a2 = cross(other_vector, (self.max - other.min))
			if cross_a1 != 0 or cross_a2 != 0:
				if cross_a1 <= 0 and cross_a2 <= 0:
					return True
				if cross_a1 >= 0 and cross_a2 >= 0:
					return False
			if cross_a1 == cross_b1 == cross_a2 == cross_b2 == 0:
				# an especially ugly special case -- lines lying on top of each other. Try to resolve instead of throwing an intersection:
				return self.min.tup < other.min.tup or (self.min.co+self.max.co).to_tuple() < (other.min.co+other.max.co).to_tuple()
			raise Intersection()

		class Sweepline:
			def __init__(self):
				self.children = blist()
			
			def add(self, item, cmp = is_below):
				low = 0; high = len(self.children)
				while low < high:
					mid = (low + high) // 2
					if cmp(self.children[mid], item):
						low = mid + 1
					else:
						high = mid
				# check for intersections
				if low > 0:
					assert cmp(self.children[low-1], item)
				if low < len(self.children):
					assert not cmp(self.children[low], item)
				self.children.insert(low, item)
			
			def remove(self, item, cmp = is_below):
				index = self.children.index(item)
				self.children.pop(index)
				if index > 0 and index < len(self.children):
					# check for intersection
					assert not cmp(self.children[index], self.children[index-1])
		
		# find edge in other and in self
		for uvedge in edge.uvedges:
			if uvedge in self.edges:
				uvedge_a = uvedge
			elif uvedge in other.edges:
				uvedge_b = uvedge
		#DEBUG
		assert uvedge_a is not uvedge_b
		
		# check if vertices and normals are aligned correctly
		verts_flipped = uvedge_b.va.vertex is uvedge_a.va.vertex
		flipped = verts_flipped ^ uvedge_a.uvface.flipped ^ uvedge_b.uvface.flipped
		# determine rotation
		first_b, second_b = (uvedge_b.va, uvedge_b.vb) if not verts_flipped else (uvedge_b.vb, uvedge_b.va)
		if not flipped:
			rot = fitting_matrix(first_b - second_b, uvedge_a.vb - uvedge_a.va)
		else:
			flip = M.Matrix(((-1,0),(0,1)))
			rot = fitting_matrix(flip * (first_b - second_b), uvedge_a.vb - uvedge_a.va) * flip
		trans = uvedge_a.vb.co - rot * first_b.co
		# extract and transform island_b's boundary
		phantoms = {uvvertex: UVVertex(rot*uvvertex.co+trans, uvvertex.vertex) for uvvertex in other.verts}
		assert uvedge_b.va in phantoms and uvedge_b.vb in phantoms
		phantoms[first_b], phantoms[second_b] = uvedge_a.vb, uvedge_a.va
		boundary_other = [UVEdge(phantoms[uvedge.va], phantoms[uvedge.vb], self) for uvedge in other.boundary_sorted if uvedge is not uvedge_b]
		# create event list
		sweepline = Sweepline()
		events_add = boundary_other + self.boundary_sorted
		events_add.remove(uvedge_a)
		events_remove = list(events_add)
		events_add.sort(key = lambda uvedge: uvedge.min.tup, reverse = True)
		events_remove.sort(key = lambda uvedge: uvedge.max.tup, reverse = True)
		try:
			while events_remove:
				while events_add and events_add[-1].min.tup <= events_remove[-1].max.tup:
					sweepline.add(events_add.pop())
				sweepline.remove(events_remove.pop())
		except Intersection:
			return False
		
		# remove edge from boundary
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
			uvedge.update()
		self.edges.update(other.edges)
		
		for uvface in other.faces:
			uvface.island = self
			uvface.verts = [phantoms[uvvertex] for uvvertex in uvface.verts]
			uvface.uvvertex_by_id = {index: phantoms[uvvertex] for index, uvvertex in uvface.uvvertex_by_id.items()}
			uvface.flipped ^= flipped
		self.faces.extend(other.faces)
		self.boundary_sorted.extend(other.boundary_sorted)
		self.boundary_sorted.sort(key = lambda uvedge: uvedge.min.tup)
		
		# everything seems to be OK
		return True

	def add(self, uvface):
		self.verts.update(uvface.verts)
		if uvface.is_sticker:
			self.stickers.append(uvface)
		else:
			self.faces.append(uvface)
	
	def generate_convex_hull(self) -> list:
		"""Returns a subset of self.verts that forms the best fitting convex polygon."""
		def make_convex_curve(verts):
			"""Remove vertices from given vert list so that the result poly is a convex curve (works for both top and bottom)."""
			result = list()
			for vertex in verts:
				while len(result) >= 2 and \
				(vertex.co-result[-1].co).to_3d().cross((result[-1].co-result[-2].co).to_3d()).z >= 0:
					result.pop()
				result.append(vertex)
			return result
		self.verts=set()
		for face in self.faces + self.stickers:
			self.verts.update(face.verts)
		verts_list = sorted(self.verts, key=lambda vertex: vertex.co.x)
		verts_top = make_convex_curve(verts_list)
		verts_bottom = make_convex_curve(reversed(verts_list))
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
		if not verts_convex:
			raise UnfoldError("Error, check topology of the mesh object (failed to calculate convex hull)")
		#go through all edges and search for the best solution
		best_score = 0
		best_box = (0, M.Vector((0,0)), M.Vector((0,0))) #(angle, box, offset) for the best score
		for vertex_a, vertex_b in pairs(verts_convex):
			if vertex_b.co == vertex_a.co:
				continue
			angle = angle2d(vertex_b.co - vertex_a.co)
			rot = M.Matrix.Rotation(angle, 2)
			#find the dimensions in both directions
			rotated = [rot*vertex.co for vertex in verts_convex]
			bottom_left = M.Vector((min(v.x for v in rotated), min(v.y for v in rotated)))
			top_right = M.Vector((max(v.x for v in rotated), max(v.y for v in rotated)))
			box = top_right - bottom_left
			score = bounding_box_score(box)
			if score > best_score:
				best_box = angle, box, bottom_left
				best_score = score
		angle, box, offset = best_box
		#switch the box so that it is taller than wider
		if box.x > box.y:
			angle += pi/2
			offset = M.Vector((-offset.y-box.y, offset.x))
			box = box.yx
		self.angle = angle
		self.bounding_box = box
		self.offset = -offset
	
	def apply_scale(self, scale=1):
		if scale != 1:
			for vertex in self.verts:
				vertex.co *= scale
	
	def save_uv(self, tex, aspect_ratio=1):
		"""Save UV Coordinates of all UVFaces to a given UV texture
		tex: UV Texture layer to use (BPy MeshUVLoopLayer struct)
		page_size: size of the page in pixels (vector)"""
		texface = tex.data
		for uvface in self.faces:
			if not uvface.is_sticker:
				rot = M.Matrix.Rotation(self.angle, 2)
				for i, uvvertex in enumerate(uvface.verts):
					uv = rot * uvvertex.co + self.pos + self.offset
					texface[uvface.face.loop_start + i].uv[0] = uv.x / aspect_ratio
					texface[uvface.face.loop_start + i].uv[1] = uv.y

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
		self.tup = tuple(self.co)
	def __hash__(self):
		if self.vertex:
			return self.vertex.index
		else:
			return hash(self.co.x) ^ hash(self.co.y)
	def __sub__(self, other):
		return self.co - other.co
	def __str__(self):
		if self.vertex:
			return "UV "+str(self.vertex.index)+" ["+strf(self.co.x)+", "+strf(self.co.y)+"]"
		else:
			return "UV * ["+strf(self.co.x)+", "+strf(self.co.y)+"]"
	def __repr__(self):
		return str(self)
	def __eq__(self, other):
		return self is other
		# originally it was: (self is other) or (self.vertex == other.vertex and self.tup == other.tup)
		# but that excludes vertices from sets in several very special cases (doubled verts that occur by coincidence)
	def __ne__(self, other):
		return not self == other
	def __lt__(self, other):
		return self.tup < other.tup
	def __le__(self, other):
		return self.tup <= other.tup
	def __ge__(self, other):
		return self.tup >= other.tup

class UVEdge:
	"""Edge in 2D"""
	def __init__(self, vertex1:UVVertex, vertex2:UVVertex, island:Island, uvface=None, edge:Edge=None):
		self.va = vertex1
		self.vb = vertex2
		self.update()
		self.island = island
		if edge:
			self.edge = edge
			edge.uvedges.append(self)
		#Every UVEdge is attached to only one UVFace. UVEdges are doubled as needed, because they both have to point clockwise around their faces
		self.uvface = uvface
	def update(self):
		"""Update data if UVVertices have moved"""
		self.min, self.max = (self.va, self.vb) if (self.va < self.vb) else (self.vb, self.va)
		y1, y2 = self.va.co.y, self.vb.co.y
		self.bottom, self.top = (y1, y2) if y1 < y2 else (y2, y1)
	def __lt__(self, other):
		return self.min.tup < other.min.tup
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
			self.flipped = False # a flipped UVFace has edges clockwise
			
			rot=z_up_matrix(face.normal)
			self.uvvertex_by_id=dict() #link vertex id -> UVVertex
			for vertex in face.verts:
				uvvertex=UVVertex(rot * vertex.co, vertex)
				self.verts.append(uvvertex)
				self.uvvertex_by_id[vertex.index]=uvvertex
			#DEBUG: check lengths
			diff=1
			for (va, uva), (vb, uvb) in pairs(zip(face.verts, self.verts)):
				diff *= (va.co-vb.co).length/(uva.co-uvb.co).length
			global differences
			differences.append((diff, face.normal))
		self.edges=list()
		edge_by_verts=dict()
		for edge in face.edges:
			edge_by_verts[(edge.va.index, edge.vb.index)]=edge
			edge_by_verts[(edge.vb.index, edge.va.index)]=edge
		for va, vb in pairs(self.verts):
			uvedge = UVEdge(va, vb, island, self, edge_by_verts[(va.vertex.index, vb.vertex.index)])
			self.edges.append(uvedge)
			island.edges.add(uvedge)
		#self.edges=[UVEdge(self.uvvertex_by_id[edge.va.data.index], self.uvvertex_by_id[edge.vb.data.index], island, edge) for edge in face.edges]
		#DEBUG:
		#self.check("construct")
	
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
		if len(self.verts) == 4:
			if (self.verts[0].co-self.verts[2].co).angle(self.verts[1].co-self.verts[3].co)<pi/10:
				print("papermodel NOTICE: This face is weirdly twisted",self.face.index, message)
		
class Sticker(UVFace):
	"""Sticker face"""
	is_sticker=True
	def __init__(self, uvedge, default_width=0.005, faces=list(), other_face=None):
		"""Sticker is directly appended to the edge given by two UVVerts"""
		#faces is a placeholder: it should contain all possibly overlaping faces
		#other_face is a placeholder too: that should be the sticking target and this sticker must fit into it
		first_vertex, second_vertex = (uvedge.va, uvedge.vb) if not uvedge.uvface.flipped else (uvedge.vb, uvedge.va)
		edge = first_vertex - second_vertex
		sticker_width=min(default_width, edge.length/2)
		other=uvedge.edge.other_uvedge(uvedge) #This is the other uvedge - the sticking target
		other_first, other_second = (other.va, other.vb) if not other.uvface.flipped else (other.vb, other.va)
		other_edge = other_second - other_first
		cos_a=cos_b=0.5 #angle a is at vertex uvedge.va, b is at uvedge.vb
		sin_a=sin_b=0.75**0.5
		len_a=len_b=sticker_width/sin_a #len_a is length of the side adjacent to vertex a, len_b similarly
		#fix overlaps with the most often neighbour - its sticking target
		if first_vertex == other_second:
			cos_a=min(max(cos_a, (edge*other_edge)/(edge.length**2)), 1) #angles between pi/3 and 0; fix for math errors
			sin_a=(1-cos_a**2)**0.5
			len_b=min(len_a, (edge.length*sin_a)/(sin_a*cos_b+sin_b*cos_a))
			if sin_a==0:
				len_a=0
			else:
				len_a=min(sticker_width/sin_a, (edge.length-len_b*cos_b)/cos_a)
		elif second_vertex == other_first:
			cos_b=min(max(cos_b, (edge*other_edge)/(edge.length**2)), 1) #angles between pi/3 and 0; fix for math errors
			sin_b=(1-cos_b**2)**0.5
			len_a=min(len_a, (edge.length*sin_b)/(sin_a*cos_b+sin_b*cos_a))
			if sin_b==0:
				len_b=0
			else:
				len_b=min(sticker_width/sin_b, (edge.length-len_a*cos_a)/cos_b)
		v3 = second_vertex.co + M.Matrix(((cos_b, -sin_b), (sin_b, cos_b))) * edge * len_b/edge.length
		v4 = first_vertex.co + M.Matrix(((-cos_a, -sin_a), (sin_a, -cos_a))) * edge * len_a/edge.length
		if v3!=v4:
			self.verts=[second_vertex, UVVertex(v3), UVVertex(v4), first_vertex]
		else:
			self.verts=[second_vertex, UVVertex(v3), first_vertex]
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
	def __init__(self, page_size_pixels:M.Vector, pure_net=True, line_thickness=1):
		"""Initialize document settings.
		page_size_pixels: document dimensions in pixels
		pure_net: if True, do not use image"""
		self.page_size = page_size_pixels
		self.scale = page_size_pixels.y
		self.pure_net = pure_net
		self.line_thickness = float(line_thickness)
	def add_mesh(self, mesh):
		"""Set the Mesh to process."""
		self.mesh=mesh
	def format_vertex(self, vector, rot=1, pos=M.Vector((0,0))):
		"""Return a string with both coordinates of the given vertex."""
		vector = rot*vector + pos
		return str(vector.x*self.scale) + " " + str((1-vector.y)*self.scale)
	def write(self, filename):
		"""Write data to a file given by its name."""
		line_through = " L ".join #utility function
		rows = "\n".join
		for num, page in enumerate(self.mesh.pages):
			with open(filename+"_"+page.name+".svg", 'w') as f:
				f.write("<?xml version='1.0' encoding='UTF-8' standalone='no'?>")
				f.write("<svg xmlns:svg='http://www.w3.org/2000/svg' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' version='1.1' width='" + str(self.page_size.x) + "px' height='" + str(self.page_size.y) + "px'>")
				f.write("""<style type="text/css">
					path {{fill:none; stroke-width:{thin:.2}px; stroke-linecap:square; stroke-linejoin:bevel; stroke-dasharray:none}}
					path.concave {{stroke:#000; stroke-dasharray:8,4,2,4; stroke-dashoffset:0}}
					path.convex {{stroke:#000; stroke-dasharray:4,8; stroke-dashoffset:0}}
					path.outer {{stroke:#000; stroke-dasharray:none; stroke-width:{thick:.2}px}}
					path.background {{stroke:#fff}}
					path.outer_background {{stroke:#fff; stroke-width:{outline:.2}px}}
					path.sticker {{fill: #fff; stroke: #000; fill-opacity: 0.4; stroke-opacity: 0.7}}
					rect {{fill:#ccc; stroke:none}}
				</style>""".format(thin=self.line_thickness, thick=1.5*self.line_thickness, outline=2*self.line_thickness))
				if not self.pure_net:
					f.write("<image x='0' y='0' width='" + str(self.page_size.x) + "' height='" + str(self.page_size.y) + "' xlink:href='file://" + filename + "_" + page.name + ".png'/>")
				if len(page.islands) > 1:
					f.write("<g>")
				for island in page.islands:
					f.write("<g>")
					rot = M.Matrix.Rotation(island.angle, 2)
					data_outer, data_convex, data_concave = list(), list(), list()
					for uvedge in island.edges:
						data_uvedge = "M " + line_through((self.format_vertex(vertex.co, rot, island.pos + island.offset) for vertex in (uvedge.va, uvedge.vb)))
						if uvedge.edge.is_cut(uvedge.uvface.face):
							assert uvedge in uvedge.uvface.island.boundary_sorted
							data_outer.append(data_uvedge)
						else:
							if uvedge.uvface.flipped ^ (uvedge.va.vertex.index > uvedge.vb.vertex.index): # each uvedge is in two opposite-oriented variants; we want to add each only once
								edge = uvedge.edge
								if edge.faces[0].uvface.flipped != edge.faces[1].uvface.flipped:
									edge.calculate_angle()
								if edge.angle > 0.01:
									data_convex.append(data_uvedge)
								elif edge.angle < -0.01:
									data_concave.append(data_uvedge)
					if island.stickers:
						data_stickers = ["<path class='sticker' d='M " + line_through((self.format_vertex(vertex.co, rot, island.pos + island.offset) for vertex in sticker.verts)) + " Z'/>" for sticker in island.stickers]
						f.write("<g>" + rows(data_stickers) + "</g>") #Stickers are separate paths in one group
					if data_outer: 
						if not self.pure_net:
							f.write("<path class='outer_background' d='" + rows(data_outer) + "'/>")
						f.write("<path class='outer' d='" + rows(data_outer) + "'/>")
					if not self.pure_net and (data_convex or data_concave):
						f.write("<path class='background' d='" + rows(data_convex + data_concave) + "'/>")
					if data_convex: f.write("<path class='convex' d='" + rows(data_convex) + "'/>")
					if data_concave: f.write("<path class='concave' d='" + rows(data_concave) + "'/>")
					f.write("</g>")
				if len(page.islands) > 1:
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
		unfolder.prepare(mark_seams=True)

		island_list = context.scene.island_list
		island_list.clear() #remove previously defined islands
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
		#	self.report(type={'ERROR_INVALID_INPUT'}, message="There are twisted quads in the model, you should divide them to triangles. Use the 'Twisted Quads' option in View Properties panel to see them.")
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
	line_thickness = bpy.props.FloatProperty(name="Line Thickness", description="SVG inner line thickness in pixels (outer lines are 1.5x thicker)", default=1, min=0, soft_max=10, subtype="UNSIGNED")
	model_scale = bpy.props.FloatProperty(name="Scale", description="Coefficient of all dimensions when exporting", default=1, soft_min=0.0001, soft_max=1.0, subtype="FACTOR")
	unfolder=None
	largest_island_ratio=0
	
	@classmethod
	def poll(cls, context):
		return context.active_object and context.active_object.type=='MESH'
	
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
		col.prop(self.properties, "bake_selected_to_active", text="Bake Selected to Active")
		layout.label(text="Document settings:")
		layout.prop(self.properties, "sticker_width")
		layout.prop(self.properties, "line_thickness")

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
		if sce.island_list:
			box.label(text="{} island(s):".format(len(sce.island_list)))
			box.template_list(sce, 'island_list', sce, 'island_list_index', rows=1, maxrows=5)
			if sce.island_list_index >= 0:
				list_item = sce.island_list[sce.island_list_index]
				box.prop(list_item, "label")
			#layout.prop(sce, "io_paper_model_display_labels", icon='RESTRICT_VIEW_OFF')
			box.prop(sce, "io_paper_model_display_islands", icon='RESTRICT_VIEW_OFF')
		else:
			box.label(text="Not unfolded")
			sub = box.row()
			sub.prop(sce, "io_paper_model_display_islands", icon='RESTRICT_VIEW_OFF')
			sub.active = False
		sub = box.row()
		sub.active = sce.io_paper_model_display_islands and bool(sce.island_list)
		sub.prop(sce, "io_paper_model_islands_alpha", slider=True)
		
		layout.operator("export_mesh.paper_model")
	
def display_islands(self, context):
	import bgl
	#TODO: save the vertex positions and don't recalculate them always
	#TODO: don't use active object, but rather save the object itself
	if context.active_object != display_islands.object:
		return
	perspMatrix = context.space_data.region_3d.perspective_matrix
	perspBuff = bgl.Buffer(bgl.GL_FLOAT, 16, [(perspMatrix[i][j]) for j in range(4) for i in range(4)])
	bgl.glLoadIdentity()
	bgl.glMatrixMode(bgl.GL_PROJECTION)
	bgl.glLoadMatrixf(perspBuff)
	bgl.glEnable(bgl.GL_BLEND)
	bgl.glBlendFunc (bgl.GL_SRC_ALPHA, bgl.GL_ONE_MINUS_SRC_ALPHA);
	bgl.glEnable(bgl.GL_POLYGON_OFFSET_FILL)
	bgl.glPolygonOffset(0, -10) #offset in Zbuffer to remove flicker
	bgl.glPolygonMode(bgl.GL_FRONT_AND_BACK, bgl.GL_FILL)
	bgl.glColor4f(1.0, 0.4, 0.0, self.io_paper_model_islands_alpha)
	ob = context.active_object
	mesh = ob.data
	global highlight_faces
	for face_id in highlight_faces:
		face = mesh.polygons[face_id]
		bgl.glBegin(bgl.GL_POLYGON)
		for vertex_id in face.vertices:
			vertex = mesh.vertices[vertex_id]
			co = vertex.co.to_4d()
			co = ob.matrix_world * co #TODO: cannot this calculation be done by GL?
			co /= co[3]
			bgl.glVertex3f(co[0], co[1], co[2])
		bgl.glEnd()
	bgl.glPolygonOffset(0.0, 0.0)
	bgl.glDisable(bgl.GL_POLYGON_OFFSET_FILL)
display_islands.handle = None
display_islands.object = None

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

def display_islands_check(scene):
	global highlight_faces
	#highlight_faces = list()
	bpy.app.handlers.scene_update_pre.remove(display_islands_check)

def list_selection_changed(self, context):
	"""Update the island highlighted in 3D View"""
	global highlight_faces
	if self.island_list_index >= 0:
		list_item = self.island_list[self.island_list_index]
		highlight_faces = [face.id for face in list_item.faces]
		display_islands.object = context.active_object
		bpy.app.handlers.scene_update_pre.append(display_islands_check)
	else:
		highlight_faces = list()
		display_islands.object = None
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
	label = bpy.props.StringProperty(name="Label", description="Label on this island", default="No Label", update=label_changed)
bpy.utils.register_class(FaceList)
bpy.utils.register_class(IslandList)

def register():
	bpy.utils.register_module(__name__)

	bpy.types.Scene.io_paper_model_display_labels = bpy.props.BoolProperty(name="Display edge priority", description="*debug property*", update=display_labels_changed)
	bpy.types.Scene.io_paper_model_display_islands = bpy.props.BoolProperty(name="Highlight selected island", update=display_islands_changed)
	bpy.types.Scene.io_paper_model_islands_alpha = bpy.props.FloatProperty(name="Highlight Alpha", description="Alpha value for island highlighting", min=0.0, max=1.0, default=0.3)
	#bpy.types.Scene.io_paper_model_display_quads = bpy.props.BoolProperty(name="Highlight tilted quads", description="Highlight tilted quad faces that would be distorted by export")
	bpy.types.Scene.island_list = bpy.props.CollectionProperty(type=IslandList, name= "Island List", description= "")
	bpy.types.Scene.island_list_index = bpy.props.IntProperty(name="Island List Index", default= -1, min= -1, max= 100, update=list_selection_changed)
	bpy.types.INFO_MT_file_export.append(menu_func)

def unregister():
	bpy.utils.unregister_module(__name__)
	bpy.types.INFO_MT_file_export.remove(menu_func)

if __name__ == "__main__":
	register()
