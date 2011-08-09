# -*- coding: utf-8 -*-
# mesh_unfold.py Copyright (C) 2010, Addam Dominec
#
# Unfolds the given mesh into a flat net and creates a print-ready document with it
#
# ***** BEGIN GPL LICENSE BLOCK *****
#
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
#
# ***** END GPL LICENCE BLOCK *****

bl_info = {
	"name": "Export Paper Model",
	"author": "Addam Dominec",
	"version": (0,7),
	"blender": (2, 5, 6),
	"api": 35302,
	"location": "File > Export > Paper Model",
	"warning": "",
	"description": "Export printable net of the selected mesh",
	"category": "Import-Export",
	"wiki_url": "http://wiki.blender.org/index.php/Extensions:2.5/Py/" \
		"Scripts/File_I-O/Paper_Model",
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
pi=3.141592653589783
priority_effect={
	'convex':1,
	'concave':1,
	'last_uncut':0.5,
	'cut_end':0,
	'last_connecting':-0.45,
	'length':-0.2}
lines=[]
twisted_quads=[]
labels=dict()
area = None

strf="{:.1f}".format
def sign(a):
	"""Return -1 for negative numbers, 1 for positive and 0 for zero."""
	if a == 0:
		return 0
	return a/abs(a)
def vectavg(vectlist):
	"""Vector average of given list."""
	if len(vectlist)==0:
		return M.Vector((0,0))
	last=vectlist.pop()
	vectlist.append(last)
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
def angle2d(direction):
	"""Get the view angle of point from origin."""
	if direction.length==0:
		return None
	if len(direction)==3:
		direction=direction.copy()
		direction.resize_2d()
	angle=direction.angle(M.Vector((1,0)))
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
def z_up_matrix(n):
	"""Get a rotation matrix that aligns given vector upwards."""
	b=n.xy.length
	l=n.length
	if b>0:
		return M.Matrix((
			(n.x*n.z/(b*l),	-n.y/b,	0),
			(n.y*n.z/(b*l),	n.x/b,	0),
			(-b/l,0,0)))
	else: #no need for rotation
		return M.Matrix((
			(1,	0,	0),
			(0,	sign(n.z),	0),
			(0,0,0)))

class UnfoldError(ValueError):
	pass

class Unfolder:
	def __init__(self, ob):
		self.ob=ob
		self.mesh=Mesh(ob.data, ob.matrix_world)

	def prepare(self, properties=None):
		"""Something that should be part of the constructor - TODO """
		self.mesh.cut_obvious()
		if not self.mesh.is_cut_enough():
			self.mesh.generate_cuts()
			#raise UnfoldError("The mesh does not have enough seams. Make it unfoldable first. (Operator \"Make Unfoldable\")")
		self.mesh.generate_islands()
		self.mesh.fix_overlaps()
		self.mesh.finalize_islands(False)
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
		self.edges_by_verts_indices={}
		self.faces=dict()
		self.islands=list()
		self.data=mesh
		self.pages=[]
		for bpy_vertex in mesh.vertices:
			self.verts[bpy_vertex.index]=Vertex(bpy_vertex, self, matrix)
		for bpy_edge in mesh.edges:
			edge=Edge(bpy_edge, self, matrix)
			self.edges[bpy_edge.index]=edge
			self.edges_by_verts_indices[(edge.va.index, edge.vb.index)]=edge
			self.edges_by_verts_indices[(edge.vb.index, edge.va.index)]=edge
		#DEBUG: this was just an idea.
		#scale = matrix.copy().invert().scale_part()
		#normatrix = M.Matrix.Scale(scale.length, 3, scale)*matrix.rotation_part()
		for bpy_face in mesh.faces:
			face=Face(bpy_face, self)#, normatrix)
			self.faces[bpy_face.index]=face
		for index in self.edges:
			self.edges[index].process_faces()
	
	def cut_obvious(self):
		"""Cut all seams and non-manifold edges."""
		count=0
		for i in self.edges:
			edge=self.edges[i]
			if not edge.is_cut() and (edge.data.use_seam or len(edge.faces)<2): #NOTE: Here is one of the rare cases when using the original BPy data (fuck this.)
				edge.cut()
				count+=1
		#DEBUG: print("There were", count, "obvious cuts.")
	
	def is_cut_enough(self, do_update_priority=False):
		"""Check for loops in the net of connected faces (that indicates the net is not unfoldable)."""
		remaining_faces=list(self.faces.values())
		while len(remaining_faces) > 0: #try all faces
			first=remaining_faces.pop()
			stack=[(first, None)] #a seed to start from
			path={first}
			while len(stack) > 0:
				#Test one island (or rather a wanna-be-island)
				current_face, previous_edge=stack.pop()
				for edge in current_face.edges:
					if edge is not previous_edge and not edge.is_cut(current_face):
						#DEBUG:
						try:
							next_face=edge.other_face[current_face]
						except KeyError:
							print ("papermodel FATAL ERROR in topology:", previous_edge, current_face, edge, edge.is_cut(), edge.is_cut(current_face), edge.data.use_seam)
						if next_face in path:
							#We've just found a loop
							#DEBUG print (edge, next_face);
							#edge.data.use_edge_sharp = True
							if do_update_priority:
								edge.priority.last_uncut() #Cutting this edge might solve the problem
							return False #if we find a loop, the mesh is not cut enough
						#else:
						stack.append((next_face, edge))
						path.add(next_face)
						remaining_faces.remove(next_face)
		#If we haven't found anything, presume there's nothing wrong
		return True
	
	def generate_cuts(self):
		"""Cut the mesh so that it will be unfoldable."""
		#Check that all quads are flat and raise an error if not
		global twisted_quads
		twisted_quads=[]
		global labels
		labels = dict()
		global area
		for index in self.faces:
			self.faces[index].check_twisted()
		#self.cut_obvious()
		count_edges_connecting = sum(not self.edges[edge_id].is_cut() for edge_id in self.edges)
		if (count_edges_connecting)==0:
			return True
		count_faces = len(self.faces)
		average_length=sum(self.edges[edge_id].length for edge_id in self.edges if not self.edges[edge_id].is_cut())/count_edges_connecting
		for edge in self.edges:
			self.edges[edge].generate_priority(average_length)
		edges_sorted=sorted(self.edges.values())
		#Iteratively cut one edge after another until it is enough
		#DEBUG print ("connecting:", count_edges_connecting, "faces:", count_faces)
		while not self.is_cut_enough(do_update_priority=True): #FIXME: terribly missing this optimisation: count_edges_connecting > count_faces or not 
			#the real thing missing is a function to detect "last connecting" and update priority, without checking cut_enough
			#I should better think it over... :/
			edge_cut=edges_sorted.pop()
			for edge in edge_cut.va.edges + edge_cut.vb.edges:
				if edge is not edge_cut and edge.va.is_in_cut(edge.vb):
					edge.priority.last_connecting() #Take down priority if cutting this edge would create a new island
					#DEBUG: 
					#edge.data.use_edge_sharp=True
				else:
					edge.priority.cut_end()
			edge_cut.cut()
			area.tag_redraw()
			if len(edges_sorted) > 0:
				edges_sorted.sort(key=lambda edge: edge.priority.value)
			count_edges_connecting -= 1
		return True
	
	def generate_islands(self):
		"""Divide faces into several Islands."""
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
		self.islands=[]
		remaining_faces=list(self.faces.values())
		#DEBUG: checking the transformation went ok
		global differences
		differences = list()
		while len(remaining_faces) > 0:
			self.islands.append(Island(remaining_faces))
		differences.sort(reverse=True)
		if differences[0][0]>10e-5:
			print ("papermodel ERROR in geometry: several faces will be deformed")
			for diff in differences[0:5]:
				print (diff)
		#FIXME: Why this?
		for edge in self.edges.values():
			edge.uvedges.sort(key=lambda uvedge: edge.faces.index(uvedge.uvface.face))
	
	def fix_overlaps(self):
		"""Split all self-overlapping Islands as needed"""
		remaining_islands = list(self.islands)
		while len(remaining_islands) > 0:
			island = remaining_islands.pop()
			uvfaces = island.get_overlap()
			if uvfaces:
				island_parts = island.split(*uvfaces)
				self.islands.remove(island)
				self.islands += island_parts
				remaining_islands += island_parts #check the new ones too
	
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
	
	def finalize_islands(self, do_split_islands=True, scale_factor=1):
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
			points=[]
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
		self.co=matrix*bpy_vertex.co
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
	def is_in_cut(self, needle):
		"""Test if both vertices are parts of the same cut tree"""
		tree_self=None
		for edge_self in self.edges:
			if edge_self.is_cut():
				for edge_needle in needle.edges:
					if edge_needle.is_cut():
						return edge_self.cut_tree is edge_needle.cut_tree
				else:
					return False
				break
		return False #just in case
class CutTree:
	"""Optimisation for finding connections in cut edges"""
	def __init__(self, edge):
		self.id=edge.data.index
		#print("New tree ", self.id)
		self.edge=edge
		self.edge_count=1
	def add(self, edge):
		"""Add an edge to this cut tree"""
		self.edge_count+=1
		edge.cut_tree=self
		pass#print ("Add", str(self.edge_count)+"th ("+str(edge.data.index)+")")
	def join(self, other):
		"""Join two (still separate) cut trees"""
		if self is not other:
			if self.edge_count <= other.edge_count:
				#print ("Join ", self.id, " (", self.edge_count, " edges) to ", other.id, " (", other.edge_count, " edges).")
				other.add(self.edge)
				path=set()
				stack=[(self.edge.va, None)]
				while len(stack) > 0:
					current, previous=stack.pop()
					path.add(current)
					for edge in current.edges:
						next = edge.other_vertex(current)
						if edge.is_cut() and next is not previous and next not in path:
							other.add(edge) #we should also call something like self.remove(edge), but that would be a waste of time
							stack.append((next, current))
			else:
				other.join(self)
		else:
			pass#print("Join myself")
class CutPriority:
	"""A container for edge's priority value"""
	def label_update(self):
		"""Debug tool"""
		global labels
		labels[self][1] = strf(self.value)
	def __init__(self, angle, rel_length):
		self.is_last_uncut=False
		self.is_last_connecting=False
		self.is_cut_end=False
		if(angle>0):
			self.value=(angle/pi)*priority_effect['convex']
		else:
			self.value=-(angle/pi)*priority_effect['concave']
		self.value+=rel_length*priority_effect['length']
	def __gt__(self, other):
		return self.value > other.value
	def __lt__(self, other):
		return self.value < other.value
	def last_connecting(self):
		"""Update priority: cutting this edge would divide an island into two."""
		if not self.is_last_connecting:
			self.value+=priority_effect['last_connecting']
			self.is_last_connecting=True
		self.label_update()
	def last_uncut(self):
		"""Update priority: this edge is one of the last that need to be cut for the mesh to be unfoldable.""" 
		if not self.is_last_uncut:
			self.value+=priority_effect['last_uncut']
			self.is_last_uncut=True
		self.label_update()
	def cut_end(self):
		"""Update priority: another edge in neighbourhood has been cut."""
		if not self.is_cut_end:
			self.value+=priority_effect['cut_end']
			self.is_cut_end=True
		self.label_update()
class Edge:
	"""Wrapper for BPy Edge"""
	def __init__(self, edge, mesh, matrix=1):
		self.data=edge
		self.va=mesh.verts[edge.vertices[0]]	
		self.vb=mesh.verts[edge.vertices[1]]
		self.vect=self.vb.co-self.va.co
		self.length=self.vect.length
		self.faces=[]
		self.angles={}
		self.other_face={} #FIXME: this should be rather a function, damn this mess
		self.uvedges=[]
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
			normal_directions={} #direction of each face's normal, rotated to 2D
			face_directions={} #direction which each face is pointing in from this edge; rotated to 2D
			is_normal_cw={}
			for face in self.faces:
				#DEBUG
				if (rot*face.normal).z > 10e-4:
					print ("papermodel ERROR in geometry, deformed face:", rot*face.normal)
				normal_directions[face]=angle2d((rot*face.normal).xy)
				face_directions[face]=angle2d((rot*(vectavg(face.verts)-self.va.co)).xy)
				is_normal_cw[face]=(normal_directions[face]-face_directions[face]) % (2*pi) < pi #True for clockwise normal around this edge, False for ccw
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
		#print ("angle: ", self.angles[self.faces[0]])
		self.priority = CutPriority(self.angles[self.faces[0]], self.length/average_length)
		labels[self.priority] = [(self.va+self.vb)*0.5, strf(self.priority.value)]
	def is_cut(self, face=None):
		"""Optional argument 'face' defines who is asking (useful for edges with more than two faces connected)"""
		#Return whether there is a cut between the two main faces
		if face is None or self.faces.index(face) <= 1:
			return self.is_main_cut
		#All other faces (third and more) are automatically treated as cut
		else:
			return True
	def cut(self):
		"""Set this edge as cut."""
		for edge_a in self.va.edges:
			if edge_a.is_cut():
				for edge_b in self.vb.edges:
					if edge_b.is_cut(): #both vertices have cut edges
						edge_b.cut_tree.join(edge_a.cut_tree)
						edge_b.cut_tree.add(self)
						break
				else: #vertex B has no cut edge (but vertex A does have)
					edge_a.cut_tree.add(self)
					break
				break
		else: #vertex A has no cut edge
			for edge_b in self.vb.edges:
				if edge_b.is_cut():
					edge_b.cut_tree.add(self)
					break
			else: #both vertices have no cut edges
				self.cut_tree=CutTree(self)
		self.data.use_seam=True #TODO: this should be optional NOTE:Here, the original BPy Edge data is used (fuck it.)
		self.is_main_cut=True
		labels[self.priority] = [(self.va+self.vb)*0.5, strf(self.priority.value)]

	def __lt__(self, other):
		"""Compare by priority."""
		return self.priority < other.priority
	def __gt__(self, other):
		"""Compare by priority."""
		return self.priority > other.priority
	def __str__(self):
		return "Edge id: "+str(self.data.index)
	def __repr__(self):
		return "Edge(id="+str(self.data.index)+"...)"
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
		#TODO: would be nice to reuse the existing normal if possible; but recalculation seems to be more stable
		self.data = bpy_face
		self.index = bpy_face.index
		self.edges = list()
		self.verts = [mesh.verts[i] for i in bpy_face.vertices]
		self.normal = (self.verts[1]-self.verts[0]).cross(self.verts[2]-self.verts[0]).normalized()
		#(bpy_face.normal*matrix).normalized()
		for verts_indices in bpy_face.edge_keys:
			edge = mesh.edges_by_verts_indices[verts_indices]
			self.edges.append(edge)
			edge.faces.append(self)
	def check_twisted(self):
		if len(self.verts) > 3:
			global twisted_quads, lines
			vert_a=self.verts[0]
			normals=[]
			for vert_b, vert_c in zip(self.verts[1:-1], self.verts[2: ]):
				normal=(vert_b.co-vert_a.co).cross(vert_c.co-vert_b.co)
				normal /= (vert_b.co-vert_a.co).length * (vert_c.co-vert_b.co).length #parallel edges have lesser weight, but lenght does not have an effect
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
	def __init__(self, faces=None, origin=None, uvfaces=None):
		"""Find an island in the given set of Faces or UVFaces.
		Note: initializing removes one island out of the given list of faces."""
		#TODO: removing from the given list is a bit clumsy
		self.faces=[]
		self.edges=[]
		self.verts=set()
		self.stickers=[]
		self.pos=M.Vector((0,0))
		self.offset=M.Vector((0,0))
		self.angle=0
		self.is_placed=False
		self.bounding_box=M.Vector((0,0))
		
		if uvfaces: #Construct from given UVFaces (typically when splitting an island into two)
			self.faces = list(uvfaces)
			for uvface in self.faces:
				self.verts.update(set(uvface.verts))
				self.edges += uvface.edges
				for uvedge in uvface.edges:
					uvedge.island = self
				uvface.island = self
		else: #Take one island from given list of Faces
			if not origin:
				#first, find where to begin
				for face in faces:
					uncut_edge_count=sum(not edge.is_cut(face) for edge in face.edges)
					if uncut_edge_count==1:
						origin=face
						break
					if uncut_edge_count==0: #single-face island
						faces.remove(face)
						self.add(UVFace(face, self))
						return
			if type(origin) is UVFace:
				self.add(origin)
				origin=origin.face
			else:
				self.add(UVFace(origin, self))
			if faces:
				faces.remove(origin)
			flood_boundary=[]
			#Create the initial wave
			for edge in origin.edges:
				if not edge.is_cut(origin):
					flood_boundary.append((origin, edge.other_face[origin], edge))
			#Spread the flood
			while len(flood_boundary)>0:
				previous_face, current_face, previous_edge=flood_boundary.pop()
				self.add(UVFace(current_face, self, previous_edge))
				if faces:
					faces.remove(current_face)
				for edge in current_face.edges:
					if edge is not previous_edge and not edge.is_cut(current_face):
						flood_boundary.append((current_face, edge.other_face[current_face], edge))
	def add(self, uvface):
		self.verts.update(set(uvface.verts))
		if uvface.is_sticker:
			self.stickers.append(uvface)
		else:
			self.faces.append(uvface)
	def generate_bounding_poly(self) -> list:
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
		verts_convex = self.generate_bounding_poly()
		#DEBUG
		if len(verts_convex)==0:
			print ("papermodel ERROR: unable to calculate convex hull")
			return M.Vector((0,0))
		#go through all edges and search for the best solution
		best_box=(0, 0, M.Vector((0,0)), M.Vector((0,0))) #(score, angle, box) for the best score
		vertex_a = verts_convex[len(verts_convex)-1]
		for vertex_b in verts_convex:
			angle=angle2d(vertex_b.co-vertex_a.co)
			if angle is not None:
				rot=M.Matrix.Rotation(angle, 2)
				#find the dimensions in both directions
				bottom_left=M.Vector((0,0))
				top_right=M.Vector((0,0))
				verts_rotated=list(map(lambda vertex: rot*vertex.co, verts_convex))
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
	def get_overlap(self) -> "(UVFace, UVFace)":
		"""Get two overlapping UVFaces of this Island"""
		#TODO: Quadratic complexity is dreadful
		for uvface in self.faces:
			overlap_uvface = uvface.get_overlap(self.faces)
			if overlap_uvface:
				return uvface, overlap_uvface
	def split(self, uvface_a, uvface_b) -> "(Island, Island)":
		"""Split this Island in half between two given UVFaces"""
		#DEBUG
		if uvface_a.is_sticker or uvface_b.is_sticker:
			raise Exception("There are stickers already")
		if uvface_a.island is not self or uvface_b.island is not self:
			raise Exception ("The faces are not mine")
		def distance(uvedge_a, uvedge_b) -> float:
			"""Distance between centers of two edges"""
			center_to_center = (uvedge_b.va.co + uvedge_b.vb.co - uvedge_a.va.co - uvedge_a.vb.co)/2
			return center_to_center.length
		island_faces = {True: set((uvface_a,)), #Faces visited from face A
			False: set((uvface_b,))} #  or we can easily get ... from face B
		flood = [ #Faces to visit next: distance, face, edge, is_from_a
			(0, uvface_a, None, True),
			(0, uvface_b, None, False)]
		while len(flood) > 0:
			prev_distance, prev_uvface, prev_uvedge, is_from_a = heappop(flood)
			for next_uvedge in prev_uvface.edges:
				if not next_uvedge.edge.is_cut(prev_uvface.face):
					next_uvface = next_uvedge.edge.other_face[prev_uvface.face].uvface
					if next_uvface in island_faces[is_from_a]:
						pass
					elif next_uvface in island_faces[not is_from_a]:
						#This edge will divide the two resulting Islands
						next_uvedge.edge.cut()
						#Just remember, return to it later
						old_va = next_uvedge.va; old_vb = next_uvedge.vb
					else:
						island_faces[is_from_a].add(next_uvface)
						if prev_uvedge:
							heappush(flood, (prev_distance + distance(prev_uvedge, next_uvedge), next_uvface, next_uvedge, is_from_a))
						else:
							heappush(flood, (prev_distance, next_uvface, next_uvedge, is_from_a))
		#Double the vertices
		va = UVVertex(old_va); vb = UVVertex(old_vb)
		for uvface in island_faces[True]:
			uvface.replace_uvvertex(old_va, va)
			uvface.replace_uvvertex(old_vb, vb)
		island_a = Island(uvfaces = island_faces[True])
		island_b = Island(uvfaces = island_faces[False])
		return island_a, island_b 
	
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
					uv = rot * uvvertex.co + self.pos + self.offset
					texface.uv_raw[2*i] = uv.x / aspect_ratio
					texface.uv_raw[2*i+1] = uv.y

class Page:
	"""Container for several Islands"""
	def __init__(self, num=1):
		self.islands=[]
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
		#quick and dirty hack for usage in sets
		if self.vertex:
			return self.vertex.index
		else:
			#DEBUG
			return 1#int(hash(self.co.x)+hash(self.co.y))
	def __eq__(self, other):
		#DEBUG
		#self.vertex==other.vertex and (self.co-other.co).length<1
		return self is other
	def __ne__(self, other):
		return not self == other
	def __str__(self):
		return "UV "+str(self.vertex.index)+" ["+strf(self.co.x)+", "+strf(self.co.y)+"]"
	def __repr__(self):
		return str(self)
class UVEdge:
	"""Edge in 2D"""
	def __init__(self, vertex1:UVVertex, vertex2:UVVertex, island:Island, uvface, edge:Edge=None):
		self.va=vertex1
		self.vb=vertex2
		self.island=island
		island.edges.append(self)
		if edge:
			self.edge=edge
			edge.uvedges.append(self)
		#Every UVEdge is attached to only one UVFace. They are doubled, if needed, because they both have to point clockwise around their faces
		self.uvface=uvface
	def __lt__(self, other):
		if type(other) is UVEdge:
			return self.edge.index < other.edge.index
		else:
			return False
	def __gt__(self, other):
		if type(other) is UVEdge:
			return self.edge.index > other.edge.index
		else:
			return True

class UVFace:
	"""Face in 2D"""
	is_sticker=False
	def __init__(self, face:Face, island:Island, fixed_edge:Edge=None):
		"""Creace an UVFace from a Face and a fixed edge.
		face: Face to take coordinates from
		island: Island to register itself in
		fixed_edge: Edge to connect to (that already has UV coordinates)"""
		#print("Create id: "+str(face.data.index))
		if type(face) is Face:
			self.verts=[]
			self.face=face
			face.uvface=self
			self.island=island
			rot=z_up_matrix(face.normal)
			self.uvvertex_by_id=dict() #link vertex id -> UVVertex
			for vertex in face.verts:
				uvvertex=UVVertex(rot*vertex.co, vertex)
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
		self.edges=[]
		edge_by_verts={}
		for edge in face.edges:
			edge_by_verts[(edge.va.index, edge.vb.index)]=edge
			edge_by_verts[(edge.vb.index, edge.va.index)]=edge
		for va, vb in pairs(self.verts):
			self.edges.append(UVEdge(va, vb, island, self, edge_by_verts[(va.vertex.index, vb.vertex.index)]))
		#self.edges=[UVEdge(self.uvvertex_by_id[edge.va.data.index], self.uvvertex_by_id[edge.vb.data.index], island, edge) for edge in face.edges]
		#DEBUG:
		self.check("construct")
		if fixed_edge:
			self.attach(fixed_edge)
	
	def __lt__(self, other):
		"""Hack for usage in heaps"""
		return self.face.index < other.face.index
	
	def __repr__(self):
		return "UVFace("+str(self.face.index)+")"
	
	def replace_uvvertex(self, current, new):
		if current in self.verts:
			self.verts[self.verts.index(current)] = new
			if current.vertex is not new.vertex:
				self.uvvertex_by_id.pop(current.vertex.index)
			self.uvvertex_by_id[new.vertex.index] = new
			for uvedge in self.edges:
				if uvedge.va == current:
					uvedge.va = new
				if uvedge.vb == current:
					uvedge.vb = new
	
	#DEBUG:
	def check(self, message=""):
		return False #UNDEBUG :)
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
		
	def attach(self, edge):
		"""Attach this face so that it sticks onto its neighbour by the given edge (thus forgetting two verts)."""
		if not edge.va.index in self.uvvertex_by_id or not edge.vb.index in self.uvvertex_by_id:
			raise ValueError("Self.verts: "+str([index for index in self.uvvertex_by_id])+" Edge.verts: "+str([edge.va.index, edge.vb.index]))
		def fitting_matrix(v1, v2):
			"""Matrix that rotates v1 to the same direction as v2"""
			return (1/pow(v1.length,2))*M.Matrix((
				(+v1.x*v2.x+v1.y*v2.y,	+v1.x*v2.y-v1.y*v2.x),
				(+v1.y*v2.x-v1.x*v2.y,	+v1.x*v2.x+v1.y*v2.y)))
		other_uvface = edge.other_face[self.face].uvface
		this_edge_vector = self.uvvertex_by_id[edge.vb.index].co-self.uvvertex_by_id[edge.va.index].co
		other_edge_vector = other_uvface.uvvertex_by_id[edge.vb.index].co-other_uvface.uvvertex_by_id[edge.va.index].co
		rot = fitting_matrix(this_edge_vector, other_edge_vector)
		offset = other_uvface.uvvertex_by_id[edge.va.index].co - rot*self.uvvertex_by_id[edge.va.index].co
		for vertex_id, uvvertex in self.uvvertex_by_id.items():
			#If this vertex is shared between both faces
			if vertex_id in other_uvface.uvvertex_by_id:
				#it's doubled, we shall share vertex data with the second face
				new_uvvertex = other_uvface.uvvertex_by_id[vertex_id]
				self.verts[self.verts.index(uvvertex)]=new_uvvertex
				self.uvvertex_by_id[vertex_id] = new_uvvertex
				for uvedge in self.edges:
					if uvedge.va is uvvertex:
						uvedge.va = new_uvvertex
					if uvedge.vb is uvvertex:
						uvedge.vb = new_uvvertex
			else: #if the vertex isn't shared, we must calculate its position
				uvvertex.co = rot*uvvertex.co + offset
		self.check("after")
	
	def get_overlap(self, others) -> "UVFace":
		"""Get a face from the given list that overlaps with this - or None if none of them overlaps."""
		#FIXME: this is bloody slow
		#FIXME: and still, it doesn't work for overlapping neighbours!
		rot = M.Matrix(((0, 1), (-1, 0)))
		for uvface in others:
			if uvface is not self:
				for this_uvedge in self.edges:
					A = this_uvedge.va.co
					B = this_uvedge.vb.co
					E = rot * (B-A)
					for other_uvedge in uvface.edges:
						if (this_uvedge.va is not other_uvedge.va and this_uvedge.vb is not other_uvedge.va and
								this_uvedge.vb is not other_uvedge.va and this_uvedge.vb is not other_uvedge.vb):
							#Check if the edges overlap
							#DEBUG:
							C = other_uvedge.va.co
							D = other_uvedge.vb.co
							F = rot * (D-C)
							try:
								a = ((A-C) * E) / ((D-C) * E)
								b = ((C-A) * F) / ((B-A) * F)
								#FIXME: when these were _ge_ and _le_, it returned more than needed - why?
								if a > 0 and a < 1 and b > 0 and b < 1:
									return uvface
							except ZeroDivisionError:
								pass
		#if len(self.verts)==3:
			#for uvface in others:
				#for uvvertex in uvface.verts:
					#if uvvertex not in self.verts and G.PointInTriangle2D(uvvertex.co, self.verts[0].co, self.verts[1].co, self.verts[2].co):
						#return uvface
		#elif len(self.verts)==4:
			#for uvface in others:
				#for uvvertex in uvface.verts:
					#if uvvertex not in self.verts and G.PointInQuad2D(uvvertex.co, self.verts[0].co, self.verts[1].co, self.verts[2].co, self.verts[3].co):
						#return uvface
		return None

class Sticker(UVFace):
	"""Sticker face"""
	is_sticker=True
	def __init__(self, uvedge, default_width=0.005, faces=[], other_face=None):
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
		v3=uvedge.vb.co+len_b*edge*(M.Matrix(((cos_b, sin_b), (-sin_b, cos_b))))/edge.length
		v4=uvedge.va.co+len_a*edge*(M.Matrix(((-cos_a, sin_a), (-sin_a, -cos_a))))/edge.length
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
		vector = rot*vector + pos
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
					#debug: bounding box
					#f.write("<rect x='"+str(island.pos.x*self.size)+"' y='"+str(self.page_size.y-island.pos.y*self.size-island.bounding_box.y*self.size)+"' width='"+str(island.bounding_box.x*self.size)+"' height='"+str(island.bounding_box.y*self.size)+"' />")
					data_outer = data_convex = data_concave = data_stickers = ""
					for uvedge in island.edges:
						data_uvedge = "\nM " + line_through([self.format_vertex(vertex.co, rot, island.pos + island.offset) for vertex in [uvedge.va, uvedge.vb]])
						#FIXME: The following clause won't return correct results for uncut edges with more than two faces connected
						if uvedge.edge.is_cut(uvedge.uvface.face):
							data_outer += data_uvedge
						else:
							if uvedge.va.vertex.index > uvedge.vb.vertex.index: #each edge is in two opposite-oriented variants; we want to add each only once
								angle = uvedge.edge.angles[uvedge.uvface.face]
								if angle > 0.01:
									data_convex += data_uvedge
								elif angle < -0.01:
									data_concave += data_uvedge
					#for sticker in island.stickers: #Stickers would be all in one path
					#	data_stickers+="\nM "+" L ".join([self.format_vertex(vertex.co, rot, island.pos+island.offset) for vertex in sticker.verts])
					#if data_stickers: f.write("<path class='sticker' d='"+data_stickers+"'/>")
					if len(island.stickers) > 0:
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
	priority_effect_convex = bpy.props.FloatProperty(name="Convex", description="Priority effect for convex edges", default=1, soft_min=-1, soft_max=10, subtype='FACTOR')
	priority_effect_concave = bpy.props.FloatProperty(name="Concave", description="Priority effect for concave edges", default=1, soft_min=-1, soft_max=10, subtype='FACTOR')
	priority_effect_last_uncut = bpy.props.FloatProperty(name="Last uncut", description="Priority effect for edges cutting of which would quicken the process", default=0.5, soft_min=-1, soft_max=10, subtype='FACTOR')
	priority_effect_cut_end = bpy.props.FloatProperty(name="Cut End", description="Priority effect for edges on ends of a cut", default=0, soft_min=-1, soft_max=10, subtype='FACTOR')
	priority_effect_last_connecting = bpy.props.FloatProperty(name="Last connecting", description="Priority effect for edges whose cutting would produce a new island", default=-0.45, soft_min=-10, soft_max=1, subtype='FACTOR')
	priority_effect_length = bpy.props.FloatProperty(name="Length", description="Priority effect of edge length (relative to object dimensions)", default=-0.2, soft_min=-10, soft_max=1, subtype='FACTOR')
	
	@classmethod
	def poll(cls, context):
		return context.active_object and context.active_object.type=="MESH"
	def execute(self, context):
		global priority_effect
		props = self.properties
		priority_effect['convex']=props.priority_effect_convex
		priority_effect['concave']=props.priority_effect_concave
		priority_effect['last_uncut']=props.priority_effect_last_uncut
		priority_effect['cut_end']=props.priority_effect_cut_end
		priority_effect['last_connecting']=props.priority_effect_last_connecting
		priority_effect['length']=props.priority_effect_length
		orig_mode=context.object.mode
		bpy.ops.object.mode_set(mode='OBJECT')
		global area
		area = context.area
		unfolder=Unfolder(context.active_object)
		unfolder.prepare()
		unfolder.mesh.data.show_edge_seams=True
		bpy.ops.object.mode_set(mode=orig_mode)
		global twisted_quads
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
	model_scale = bpy.props.FloatProperty(name="Scale", description="Coefficient of all dimensions when exporting", default=1, soft_min=0.001, soft_max=10, subtype="FACTOR")
	unfolder=None
	largest_island_ratio=0
	
	@classmethod
	def poll(cls, context):
		return context.active_object and context.active_object.type=="MESH"
	
	def execute(self, context):
		try:
			self.unfolder.save(self.properties)
			return {"FINISHED"}
		except UnfoldError as error:
			self.report(type="ERROR_INVALID_INPUT", message=error.args[0])
			return {"CANCELLED"}
		except:
			raise
	
	def invoke(self, context, event):
		sce=context.scene
		self.properties.bake_selected_to_active = sce.render.use_bake_selected_to_active
		
		self.unfolder=Unfolder(context.active_object)
		self.unfolder.prepare(self.properties)
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
		scale_ratio = self.unfolder.mesh.largest_island_ratio(M.Vector((self.properties.output_size_x, self.properties.output_size_y))) * self.properties.model_scale
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
	bl_region_type = "UI"
	handles = dict()
	def display_quads(self, context):
		import bgl, blf
		perspMatrix = context.space_data.region_3d.perspective_matrix
		perspBuff = bgl.Buffer(bgl.GL_FLOAT, 16, [perspMatrix[i][j] for i in range(4) for j in range(4)])
		bgl.glLoadIdentity()
		bgl.glMatrixMode(bgl.GL_PROJECTION)
		bgl.glLoadMatrixf(perspBuff)
		
		"""bgl.glColor4f(1,0.2,0,0.2)
		blf.position(0, 1, 1, 1)
		blf.aspect(0,0.005)
		blf.size(0,12,150)
		blf.draw(0, "Hello, world!")
		blf.aspect(0, 1)"""
		global lines, twisted_quads
		for s, e in lines:
			bgl.glBegin(bgl.GL_LINE_STRIP)
			bgl.glColor4f(1,0.2,0,0.7)
			bgl.glVertex3f(s[0], s[1], s[2])
			bgl.glVertex3f(e[0], e[1], e[2])
			
			bgl.glEnd()
		bgl.glEnable(bgl.GL_BLEND)
		bgl.glBlendFunc (bgl.GL_SRC_ALPHA, bgl.GL_ONE_MINUS_SRC_ALPHA);
		bgl.glColor4f(1,0.4,0,0.3)
		for quad in twisted_quads:
			bgl.glBegin(bgl.GL_POLYGON)
			for vertex in quad:
				bgl.glVertex3f(vertex.co[0], vertex.co[1], vertex.co[2])
			bgl.glEnd()

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
			vec = view_mat * position
			vec /= vec[3]
			x = int(mid_x + vec[0]*width/2.0)
			y = int(mid_y + vec[1]*height/2.0)
			blf.position(0, x, y, 0)
			blf.draw(0, label)
	display_labels.handle = None

	def draw(self, context):
		layout = self.layout
		layout.prop(context.scene, "io_paper_model_display_labels", icon='RESTRICT_VIEW_OFF')
		"""global paper_model_handles
		region=[region for region in context.area.regions if region.type=='WINDOW'][0]
		for name in paper_model_handles:
			value, mode = paper_model_handles[name]
			property_name = "io_paper_model_"+name
			layout.prop(context.scene, property_name, icon='RESTRICT_VIEW_OFF')
			if value is None and getattr(context.scene, property_name):
				paper_model_handles[name]=region.callback_add(getattr(self, name), (context,), mode), mode
				context.area.tag_redraw()
			elif value is not None and not getattr(context.scene, property_name):
				region.callback_remove(value)
				paper_model_handles[name]=None, mode
				context.area.tag_redraw()"""
	
def prop_changed(self, context):
	region = [region for region in context.area.regions if region.type=='WINDOW'][0]
	display_labels = VIEW3D_PT_paper_model.display_labels
	if self.io_paper_model_display_labels:
		display_labels.handle = region.callback_add(display_labels, (self, context), "POST_PIXEL")
	else:
		region.callback_remove(display_labels.handle)

def register():
	bpy.utils.register_module(__name__)

	bpy.types.Scene.io_paper_model_display_labels = bpy.props.BoolProperty(attr="io_paper_model_display_labels", name="Display labels", update=prop_changed)
	bpy.types.Scene.io_paper_model_display_quads = bpy.props.BoolProperty(name="Highlight tilted quads", description="*not working* Highlight tilted quad faces that would be distorted by export")
	bpy.types.INFO_MT_file_export.append(menu_func)
	#bpy.types.register(VIEW3D_PT_paper_model)

def unregister():
#	bpy.types.unregister(VIEW3D_PT_paper_model)
	bpy.utils.unregister_module(__name__)
	bpy.types.INFO_MT_file_export.remove(menu_func)

if __name__ == "__main__":
	register()
