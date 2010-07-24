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

bl_addon_info = {
    'name': 'Export: Paper Model',
    'author': 'Addam Dominec',
    'version': '0.6',
    'blender': (2, 5, 3),
    'location': 'View3D > Toolbox > Unfold',
    'description': 'Export printable net of a given mesh',
    'category': 'Import/Export',
    'wiki_url': 'http://wiki.blender.org/index.php/Extensions:2.5/Py/Scripts/File_I-O/Paper_Model',
    'tracker_url': 'https://projects.blender.org/tracker/index.php?func=detail&aid=22417&group_id=153&atid=467'}

"""

Additional links:
    e-mail: adominec {at} gmail {dot} com

"""
import bpy
import mathutils as M
import geometry as G
import time
pi=3.141592653589783
priority_effect={
	'convex':1,
	'concave':1,
	'last_uncut':0.5,
	'cut_end':0,
	'last_connecting':-0.45,
	'length':-0.2}

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
	sum=last.co.copy().zero() #keep the dimensions
	for vect in vectlist:
		sum+=vect.co
	return sum/vectlist.__len__()
def angle2d(direction):
	"""Get the view angle of point from origin."""
	if direction.length==0:
		return None
	angle=direction.angle(M.Vector((1,0)))
	if direction[0]<0:
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

class UnfoldError(ValueError):
	pass

class Unfolder:
	page_size=M.Vector((400,600))
	def __init__(self, ob):
		self.ob=ob
		self.mesh=Mesh(ob.data, ob.matrix_world)
	def unfold(self):
		"""Decide for seams and preview them to the user."""
		self.mesh.generate_cuts()
	def save(self, properties):
		"""Export the document."""
		filepath=properties.filepath
		if filepath[-4:]==".svg" or filepath[-4:]==".png":
			filepath=filepath[0:-4]
		self.page_size=M.Vector((properties.output_size_x, properties.output_size_y)) #in meters
		scale_length=bpy.context.scene.unit_settings.scale_length
		self.size=properties.output_dpi*scale_length*100/2.54 #in points per meter
		self.mesh.cut_obvious()
		if not self.mesh.is_cut_enough():
			self.unfold()
			#raise UnfoldError("The mesh does not have enough seams. Make it unfoldable first. (Operator \"Make Unfoldable\")")
		self.mesh.generate_islands()
		self.mesh.generate_stickers(properties.sticker_width)
		self.mesh.fit_islands(self.page_size)
		if not properties.output_pure:
			self.mesh.save_uv(page_size=self.page_size)
			#this property might be overriden when invoking the Operator
			selected_to_active=bpy.context.scene.render.bake_active; bpy.context.scene.render.bake_active=properties.bake_selected_to_active
			self.mesh.save_image(filepath, self.page_size*self.size)
			#revoke settings
			bpy.context.scene.render.bake_active=selected_to_active
		svg=SVG(self.size, self.page_size)
		svg.add_mesh(self.mesh)
		svg.write(filepath)

class Mesh:
	"""Wrapper for Bpy Mesh"""
	def __init__(self, mesh, matrix):
		self.verts=dict()
		self.edges=dict()
		self.faces=dict()
		self.edges_sorted=list()
		self.islands=list()
		self.data=mesh
		self.pages=[]
		for vertex in mesh.verts:
			self.verts[vertex.index]=Vertex(vertex, self, matrix)
		for edge in mesh.edges:
			self.edges[edge.index]=Edge(edge, self)
		for face in mesh.faces:
			self.faces[face.index]=Face(face, self)
	def cut_obvious(self):
		"""Cut all seams and non-manifold edges."""
		for i in self.edges:
			edge=self.edges[i]
			if edge.data.seam or edge.fa is None or edge.fb is None:
				edge.cut()
	def is_cut_enough(self, do_update_priority=False):
		"""Check for loops in the net of connected faces (that indicates the net is not unfoldable)."""
		remaining_faces=list(self.faces.values())
		while len(remaining_faces) > 0: #try all faces
			first=remaining_faces.pop()
			stack=[(first, None)] #a seed to start from
			path={first}
			while len(stack) > 0:
				#test one island
				current, previous=stack.pop()
				for edge in current.edges:
					if not edge.is_cut:
						next=edge.other_face(current)
						if next is not previous:
							if next in path:
								if do_update_priority:
									edge.priority.last_uncut() #if this edge might solve the problem
								return False #if we find a loop, the mesh is not cut enough
							stack.append((next, current))
							path.add(next)
							remaining_faces.remove(next)
		#If we haven't found anything, presume there's nothing wrong
		return True
	def generate_cuts(self):
		"""Cut the mesh so that it will be unfoldable."""
		self.cut_obvious()
		count_edges_connecting = sum(not self.edges[edge_id].is_cut for edge_id in self.edges)
		count_faces = len(self.faces)
		average_length=sum(self.edges[edge_id].length for edge_id in self.edges if not self.edges[edge_id].is_cut)/count_edges_connecting
		for edge in self.edges:
			self.edges[edge].generate_priority(average_length)
		self.edges_sorted=sorted(self.edges.values())
		#Iteratively cut one edge after another until it is enough
		while count_edges_connecting > count_faces or not self.is_cut_enough(do_update_priority=True):
			edge_cut=self.edges_sorted.pop()
			for edge in edge_cut.va.edges + edge_cut.vb.edges:
				if edge is not edge_cut and edge.va.is_in_cut(edge.vb):
					edge.priority.last_connecting() #Take down priority if cutting this edge would create a new island
					edge.data.sharp=True
				else:
					edge.priority.cut_end()
			edge_cut.cut()
			if len(self.edges_sorted) > 0:
				self.edges_sorted.sort(key=lambda edge: edge.priority.value)
			count_edges_connecting -= 1
		return True
	def generate_islands(self):
		"""Divide faces into several Islands."""
		def connected_faces(border_edges, inner_faces):
			outer_faces=list()
			for edge in border_edges:
				if edge.fa and not edge.fa in inner_faces and not edge.fa in outer_faces:
					outer_faces.append(edge.fa)
				if edge.fb and not edge.fb in inner_faces and not edge.fb in outer_faces:
					outer_faces.append(edge.fb)
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
		while len(remaining_faces) > 0:
			self.islands.append(Island(remaining_faces))
	def generate_stickers(self, default_width):
		"""Add sticker faces where they are needed."""
		#TODO: it should take into account overlaps with faces and with already created stickers and size of the face that sticker will be actually sticked on
		def uvedge_priority(uvedge):
			"""Retuns whether it is a good idea to create a sticker on this edge"""
			#This is just a placeholder
			return uvedge.va.co.y
		for edge_index in self.edges:
			edge=self.edges[edge_index]
			if edge.is_cut and len(edge.uvedges)==2:
				if uvedge_priority(edge.uvedges[0]) >= uvedge_priority(edge.uvedges[1]):
					edge.uvedges[0].island.stickers.append(Sticker(edge.uvedges[0], default_width))
				else:
					edge.uvedges[1].island.stickers.append(Sticker(edge.uvedges[1], default_width))
	def fit_islands(self, page_size):
		"""Move islands so that they fit into pages, based on their bounding boxes"""
		#this algorithm is not optimal, but cool enough
		#it handles with two basic domains:
		#list of Points: they describe all sensible free rectangle area available on the page
		#Boundaries: linked list of points around the used area and the page - makes many calculations a lot easier
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
		islands=list(self.islands)
		largestErrorRatio=1
		for island in islands:
			island.generate_bounding_box()
			if island.bounding_box.x > page_size.x or island.bounding_box.y > page_size.y:
				errorRatio=max(island.bounding_box.x/page_size.x, island.bounding_box.y/page_size.y)
				if errorRatio > largestErrorRatio:
					largestErrorRatio=errorRatio
		if largestErrorRatio > 1:
			raise UnfoldError("An island is too big to fit to the page size. To make the export possible, scale the object down "+str(largestErrorRatio)[:4]+" times.")
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
	def save_uv(self, page_size=M.Vector((744, 1052))): #page_size is in pixels
		bpy.ops.object.mode_set()
		bpy.ops.mesh.uv_texture_add()
		#note: expecting that the active object's data is self.mesh
		tex=self.data.active_uv_texture
		tex.name="Unfolded"
		for island in self.islands:
			island.save_uv(tex, page_size)
	def save_image(self, filename, page_size=M.Vector((744, 1052))): #page_size is in pixels
		rd=bpy.context.scene.render
		recall_margin=rd.bake_margin; rd.bake_margin=0
		recall_clear=rd.bake_clear; rd.bake_clear=False
		for page in self.pages:
			#image=bpy.data.images.new(name="Unfolded "+self.data.name+" "+page.name, width=int(page_size.x), height=int(page_size.y))
			image_name="Unfolded "+self.data.name+" "+page.name
			obstacle=bpy.data.images.get(image_name)
			if obstacle:
				obstacle.name=image_name[0:-1] #when we create the new image, we want it to have *exactly* the name we assign
			bpy.ops.image.new(name=image_name, width=int(page_size.x), height=int(page_size.y), color=(1,1,1))
			image=bpy.data.images.get(image_name) #this time it is our new image
			image.filepath_raw=filename+"_"+page.name+".png"
			image.file_format='PNG'
			texfaces=self.data.active_uv_texture.data
			for island in page.islands:
				for uvface in island.faces:
					if not uvface.is_sticker:
						texfaces[uvface.face.data.index].image=image
			bpy.ops.object.bake_image()
			image.save()
		rd.bake_margin=recall_margin
		rd.bake_clear=recall_clear
   
class Vertex:
	"""BPy Vertex wrapper"""
	def __init__(self, vertex, mesh=None, matrix=1):
		self.data=vertex
		self.co=matrix*vertex.co
		self.edges=list()
		self.uvs=list()
	def __hash__(self):
		return hash(self.data.index)
	def __eq__(self, other):
		if type(other) is type(self):
			return self.data.index==other.data.index
		else:
			return False
	def is_in_cut(self, needle):
		"""Test if both vertices are parts of the same cut tree"""
		time_begin=time.clock()
		tree_self=None
		for edge_self in self.edges:
			if edge_self.is_cut:
				for edge_needle in needle.edges:
					if edge_needle.is_cut:
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
						if edge.is_cut and next is not previous and next not in path:
							other.add(edge) #we should also call something like self.remove(edge), but that would be a waste of time
							stack.append((next, current))
			else:
				other.join(self)
		else:
			pass#print("Join myself")
class CutPriority:
	"""A container for edge's priority value"""
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
	def last_uncut(self):
		"""Update priority: this edge is one of the last that need to be cut for the mesh to be unfoldable.""" 
		if not self.is_last_uncut:
			self.value+=priority_effect['last_uncut']
			self.is_last_uncut=True
	def cut_end(self):
		"""Update priority: another edge in neighbourhood has been cut."""
		if not self.is_cut_end:
			self.value+=priority_effect['cut_end']
			self.is_cut_end=True
class Edge:
	"""Wrapper for BPy Edge"""
	def __init__(self, edge, mesh):
		self.data=edge
		self.va=mesh.verts[edge.verts[0]]	
		self.vb=mesh.verts[edge.verts[1]]
		self.vect=self.vb.co-self.va.co
		self.length=self.vect.length #FIXME: must take the object's matrix into account
		self.fa=None
		self.fb=None
		self.uvedges=[]
		self.is_cut=False
		self.priority=None
		self.angle=0
		self.va.edges.append(self)
		self.vb.edges.append(self)
	def generate_priority(self, average_length=1):
		"""Calculate initial priority value."""
		self.priority=CutPriority(self.angle, self.length/average_length)
	def cut(self):
		"""Set this edge as cut."""
		for edge_a in self.va.edges:
			if edge_a.is_cut:
				for edge_b in self.vb.edges:
					if edge_b.is_cut: #both vertices have cut edges
						edge_b.cut_tree.join(edge_a.cut_tree)
						edge_b.cut_tree.add(self)
						break
				else: #vertex B has no cut edge (but vertex A does have)
					edge_a.cut_tree.add(self)
					break
				break
		else: #vertex A has no cut edge
			for edge_b in self.vb.edges:
				if edge_b.is_cut:
					edge_b.cut_tree.add(self)
					break
			else: #both vertices have no cut edges
				self.cut_tree=CutTree(self)
		self.data.seam=True #TODO: this should be optional
		self.is_cut=True
	def __lt__(self, other):
		"""Compare by priority."""
		return self.priority < other.priority
	def __gt__(self, other):
		"""Compare by priority."""
		return self.priority > other.priority
	def other_vertex(self, this):
		"""Get a vertex of this edge that is not the given one - or None if none of both vertices is the given one."""
		if self.va is this:
			return self.vb
		elif self.vb is this:
			return self.va
		return None
	def other_face(self, this):
		"""Get a face of this edge that is not the given one - or None if none of both faces is the given one."""
		if this is self.fa:
			return self.fb
		elif this is self.fb:
			return self.fa
		else:
			raise ValueError("Edge "+str(self.data.index)+" has faces "+str([self.fa.data.index, self.fb.data.index])+", but not "+str(this.data.index))
	def other_uvedge(self, this):
		"""Get an uvedge of this edge that is not the given one - or None if no other uvedge was found."""
		for uvedge in self.uvedges:
			if uvedge is not this:
				return uvedge
		else:
			return None
class Face:
	"""Wrapper for BPy Face"""
	def __init__(self, face, mesh):
		self.data=face
		self.edges=list()
		self.verts=[mesh.verts[i] for i in face.verts]
		for vertex in face.verts:
			for edge in mesh.verts[vertex].edges:
				if not edge in self.edges and \
						edge.va in self.verts and edge.vb in self.verts:
					self.edges.append(edge)
					if edge.fa==None:
						edge.fa=self
					else:
						edge.fb=self
						is_convex=sign((edge.va.co-vectavg(self.verts))*edge.fa.data.normal)
						edge.angle=face.normal.angle(edge.fa.data.normal)*is_convex
	def __hash__(self):
		return hash(self.data.index)
	def flatten_matrix(self):
		"""Get a rotation matrix that aligns this face with the ground."""
		n=self.data.normal
		b=n.xy.length
		l=n.length
		if b>0:
			return M.Matrix(
				(n.x*n.z/(b*l),	-n.y/b),
				(n.y*n.z/(b*l),	n.x/b),
				(-b/l,					0))
		else: #no need for rotation
			return M.Matrix(
				(1,	0),
				(0,	sign(n.z)),
				(0,	0))
class Island:
	def __init__(self, faces):
		"""Find an island in the given set of faces.
		Note: initializing takes one island out of the given list of faces."""
		self.faces=[]
		self.edges=[]
		self.stickers=[]
		self.pos=M.Vector((0,0))
		self.offset=M.Vector((0,0))
		self.angle=0
		self.is_placed=False
		self.bounding_box=M.Vector((0,0))
		#first, find where to begin
		origin=False
		for face in faces:
			uncut_edge_count=sum(not edge.is_cut for edge in face.edges)
			if uncut_edge_count==1:
				origin=face
				faces.remove(origin)
				break
			if uncut_edge_count==0: #single-face island
				faces.remove(face)
				self.faces.append(UVFace(face, self))
				break
		if origin:
			self.faces.append(UVFace(origin, self))
			flood_boundary=list()
			for edge in origin.edges:
				if not edge.is_cut:
					flood_boundary.append((origin, edge.other_face(origin), edge))
			while len(flood_boundary)>0:
				origin, current_face, current_edge=flood_boundary.pop()
				self.faces.append(UVFace(current_face, self, current_edge))
				faces.remove(current_face)
				for edge in current_face.edges:
					if edge is not current_edge and not edge.is_cut:
						flood_boundary.append((current_face, edge.other_face(current_face), edge))
	def generate_bounding_poly(self):
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
		verts=set()
		for face in self.faces + self.stickers:
			verts=verts.union(face.verts)
		verts_top=list(verts)
		verts_top.sort(key=lambda vertex: vertex.co.x) #sorted left to right
		make_convex_curve(verts_top)
		verts_bottom=list(verts)
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
		if len(verts_convex)==0:
			print ("Fuck it.")
			return M.Vector((0,0))
		#go through all edges and search for the best solution
		best_box=(0, 0, M.Vector((0,0)), M.Vector((0,0))) #(score, angle, box) for the best score
		vertex_a = verts_convex[len(verts_convex)-1]
		for vertex_b in verts_convex:
			angle=angle2d(vertex_b.co-vertex_a.co)
			if angle is not None:
				rot=M.RotationMatrix(angle, 2)
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
	def save_uv(self, tex, page_size=M.Vector((744, 1052))): #page_size is in pixels
		"""Save UV Coordinates of all UVFaces to a given UV texture
		tex: UV Texture layer to use (BPy MeshTextureFaceLayer struct)
		page_size: size of the page in pixels (vector)"""
		for face in self.faces:
			if not face.is_sticker:
				texface=tex.data[face.face.data.index]
				rot=M.RotationMatrix(self.angle, 2)
				for i in range(len(face.verts)):
					uv=rot*face.verts[i].co+self.pos+self.offset
					texface.uv_raw[2*i]=uv.x/page_size.x
					texface.uv_raw[2*i+1]=uv.y/page_size.y

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
		self.co=(M.Vector(vector)).resize2D()
		self.vertex=vertex
	def __hash__(self):
		#quick and dirty hack for usage in sets
		return int(hash(self.co.x)*10000000000+hash(self.co.y))
	def __eq__(self, other):
		return self.vertex==other.vertex and (self.co-other.co).length<0.00001
	def __ne__(self, other):
		return not self==other
	def __str__(self):
		return "UV ["+str(self.co.x)[0:6]+", "+str(self.co.y)[0:6]+"]"
	def __repr__(self):
		return str(self)
class UVEdge:
	"""Edge in 2D"""
	def __init__(self, vertex1, vertex2, island=None, edge=None):
		self.va=vertex1
		self.vb=vertex2
		self.island=island
		island.edges.append(self)
		if edge:
			self.edge=edge
			edge.uvedges.append(self)
class UVFace:
	"""Face in 2D"""
	is_sticker=False
	def __init__(self, face, island=None, fixed_edge=None):
		"""Creace an UVFace from a Face and a fixed edge.
		face: Face to take coordinates from
		island: Island to register itself in
		fixed_edge: Edge to connect to (that already has UV coordinates)"""
		#print("Create id: "+str(face.data.index))
		if type(face) is Face:
			self.verts=[]
			self.face=face
			face.uvface=self
			self.island=None
			rot=face.flatten_matrix()
			self.uvvertex_by_id=dict() #link vertex id -> UVVertex
			for vertex in face.verts:
				uvvertex=UVVertex(rot*vertex.co, vertex)
				self.verts.append(uvvertex)
				self.uvvertex_by_id[vertex.data.index]=uvvertex
		elif type(face) is UVFace: #copy constructor
			self.verts=list(face.verts)
			self.face=face.face
			self.uvvertex_by_id=dict(face.uvvertex_by_id)
		self.edges=[]
		edge_by_verts={}
		for edge in face.edges:
			edge_by_verts[(edge.va.data.index, edge.vb.data.index)]=edge
			edge_by_verts[(edge.vb.data.index, edge.va.data.index)]=edge
		for va, vb in pairs(self.verts):
			self.edges.append(UVEdge(va, vb, island, edge_by_verts[(va.vertex.data.index, vb.vertex.data.index)]))
		#self.edges=[UVEdge(self.uvvertex_by_id[edge.va.data.index], self.uvvertex_by_id[edge.vb.data.index], island, edge) for edge in face.edges]
		if fixed_edge:
			self.attach(fixed_edge)
	def attach(self, edge):
		"""Attach this face so that it sticks onto its neighbour by the given edge (thus forgetting two verts)."""
		if not edge.va.data.index in self.uvvertex_by_id or not edge.vb.data.index in self.uvvertex_by_id:
			raise ValueError("Self.verts: "+str([index for index in self.uvvertex_by_id])+" Edge.verts: "+str([edge.va.data.index, edge.vb.data.index]))
		def fitting_matrix(v1, v2):
			"""Matrix that rotates v1 to the same direction as v2"""
			return (1/pow(v1.length,2))*M.Matrix(
				(+v1.x*v2.x+v1.y*v2.y,	+v1.x*v2.y-v1.y*v2.x),
				(+v1.y*v2.x-v1.x*v2.y,	+v1.x*v2.x+v1.y*v2.y))
		other_face=edge.other_face(self.face).uvface
		this_edge=self.uvvertex_by_id[edge.vb.data.index].co-self.uvvertex_by_id[edge.va.data.index].co
		other_edge=other_face.uvvertex_by_id[edge.vb.data.index].co-other_face.uvvertex_by_id[edge.va.data.index].co
		rot=fitting_matrix(this_edge, other_edge)
		offset=other_face.uvvertex_by_id[edge.va.data.index].co-rot*self.uvvertex_by_id[edge.va.data.index].co
		for vertex_id in self.uvvertex_by_id:
			vertex=self.uvvertex_by_id[vertex_id]
			vertex.co=rot*vertex.co+offset
			if vertex_id in other_face.uvvertex_by_id: #means that this is a shared vertex; it's doubled, we shall share vertex data with the second face
				self.verts[self.verts.index(self.uvvertex_by_id[vertex_id])]=other_face.uvvertex_by_id[vertex_id]
				self.uvvertex_by_id[vertex_id]=other_face.uvvertex_by_id[vertex_id]
			#else: #if the vertex isn't shared, we must calculate its position
		for uvedge in edge.uvedges:
			if uvedge in self.edges:
				this_uvedge=uvedge
			else:
				other_uvedge=uvedge
		this_uvedge.va=other_uvedge.vb
		this_uvedge.vb=other_uvedge.va
	def get_overlap(self, others):
		"""Get a face from the given list that overlaps with this - or None if none of them overlaps."""
		if len(self.verts)==3:
			for face in others:
				for vertex in face.verts:
					if G.PointInTriangle2D(vertex.co, self.verts[0].co, self.verts[1].co, self.verts[2].co):
						return face
		elif len(self.verts)==4:
			for face in others:
				for vertex in face.verts:
					if G.PointInQuad2D(vertex.co, self.verts[0].co, self.verts[1].co, self.verts[2].co, self.verts[3].co):
						return face
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
		v3=uvedge.vb.co+len_b*(M.Matrix((cos_b, sin_b), (-sin_b, cos_b))*edge)/edge.length
		v4=uvedge.va.co+len_a*(M.Matrix((-cos_a, sin_a), (-sin_a, -cos_a))*edge)/edge.length
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
	def __init__(self, size=1, page_size=M.Vector((0.210, 0.297))):
		"""Initialize document settings.
		size: factor to all vertex coordinates (float)
		page_size: document dimensions in meters (vector)"""
		self.size=size
		self.page_size=page_size
	def add_mesh(self, mesh):
		"""Set the Mesh to process."""
		self.mesh=mesh
	def format_vertex(self, vector, rot=1, pos=M.Vector((0,0))):
		"""Return a string with both coordinates of the given vertex."""
		vector=rot*vector+pos
		return str(vector.x*self.size)+" "+str((self.page_size.y-vector.y)*self.size)
	def write(self, filename):
		"""Write data to a file given by its name."""
		is_image=not bpy.context.scene.unfolder_output_pure
		for num, page in enumerate(self.mesh.pages):
			with open(filename+"_"+page.name+".svg", 'w') as f:
				f.write("<?xml version='1.0' encoding='UTF-8' standalone='no'?>")
				f.write("<svg xmlns:svg='http://www.w3.org/2000/svg' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' version='1.1' width='"+str(self.page_size.x*self.size)+"px' height='"+str(self.page_size.y*self.size)+"px'>")
				f.write("""<style type="text/css">
					path {fill:none; stroke-width:1px; stroke-linecap:square; stroke-linejoin:bevel; stroke-dasharray:none}
					path.concave {stroke:#000; stroke-dasharray:4,8; stroke-dashoffset:0}
					path.convex {stroke:#000; stroke-dasharray:8,4,2,4; stroke-dashoffset:0}
					path.outer {stroke:#000; stroke-dasharray:none; stroke-width:1.5px}
					path.background {stroke:#fff}
					path.outer_background {stroke:#fff; stroke-width:2px}
					path.sticker {fill: #fff; stroke: #000; fill-opacity: 0.4; stroke-opacity: 0.7}
					rect {fill:#ccc; stroke:none}
				</style>""")
				if is_image:
					f.write("<image x='0' y='0' width='"+str(self.page_size.x*self.size)+"' height='"+str(self.page_size.y*self.size)+"' xlink:href='file://"+filename+"_"+page.name+".png'/>")
				f.write("<g>")
				for island in page.islands:
					f.write("<g>")
					rot=M.RotationMatrix(island.angle, 2)
					#debug: bounding box
					#f.write("<rect x='"+str(island.pos.x*self.size)+"' y='"+str(self.page_size.y-island.pos.y*self.size-island.bounding_box.y*self.size)+"' width='"+str(island.bounding_box.x*self.size)+"' height='"+str(island.bounding_box.y*self.size)+"' />")
					line_through=" L ".join
					data_outer=""
					data_convex=""
					data_concave=""
					data_stickers=""
					for uvedge in island.edges:
						data_uvedge="\nM "+line_through([self.format_vertex(vertex.co, rot, island.pos+island.offset) for vertex in [uvedge.va, uvedge.vb]])
						if uvedge.edge.is_cut:
							data_outer+=data_uvedge
						else:
							if uvedge.va.vertex.data.index > uvedge.vb.vertex.data.index: #each edge is in two opposite-oriented variants; we want to add each only once
								if uvedge.edge.angle>0:
									data_convex+=data_uvedge
								elif uvedge.edge.angle<0:
									data_concave+=data_uvedge
					#for sticker in island.stickers: #Stickers would be all in one path
					#	data_stickers+="\nM "+" L ".join([self.format_vertex(vertex.co, rot, island.pos+island.offset) for vertex in sticker.verts])
					#if data_stickers: f.write("<path class='sticker' d='"+data_stickers+"'/>")
					if len(island.stickers)>0:
						f.write("<g>")
						for sticker in island.stickers: #Stickers are separate paths in one group
							f.write("<path class='sticker' d='M "+line_through([self.format_vertex(vertex.co, rot, island.pos+island.offset) for vertex in sticker.verts])+" Z'/>")
						f.write("</g>")
					if data_outer: 
						if is_image: f.write("<path class='outer_background' d='"+data_outer+"'/>")
						f.write("<path class='outer' d='"+data_outer+"'/>")
					if is_image and (data_convex or data_concave): f.write("<path class='background' d='"+data_convex+data_concave+"'/>")
					if data_convex: f.write("<path class='convex' d='"+data_convex+"'/>")
					if data_concave: f.write("<path class='concave' d='"+data_concave+"'/>")
					f.write("</g>")
				f.write("</g>")
				f.write("</svg>")
				f.close()

class MESH_OT_make_unfoldable(bpy.types.Operator):
	"""Blender Operator: unfold the selected object."""
	bl_idname = "MESH_OT_make_unfoldable"
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
	def poll(self, context):
		return context.active_object and context.active_object.type=="MESH"
	def execute(self, context):
		global priority_effect
		props = self.properties
		priority_effect['convex']=props.priority_effect_convex
		priority_effect['concave']=props.priority_effect_convex
		priority_effect['last_uncut']=props.priority_effect_last_uncut
		priority_effect['cut_end']=props.priority_effect_cut_end
		priority_effect['last_connecting']=props.priority_effect_last_connecting
		priority_effect['length']=props.priority_effect_length
		orig_mode=context.object.mode
		"""print(props.edit)
		if props.edit:
			bpy.ops.object.mode_set(mode='EDIT')
			bpy.ops.mesh.select_all(action='SELECT')
			print("Zvláštní...")
			bpy.ops.mesh.mark_seam(clear=True)
			bpy.ops.mesh.select_all(action='TOGGLE')
			print("Až sem vše v pořádku")"""
		bpy.ops.object.mode_set(mode='OBJECT')
		unfolder=Unfolder(context.active_object)
		unfolder.unfold()
		unfolder.mesh.data.draw_seams=True
		bpy.ops.object.mode_set(mode=orig_mode)
		return {'FINISHED'}

class EXPORT_OT_paper_model(bpy.types.Operator):
	"""Blender Operator: save the selected object's net and optionally bake its texture"""
	bl_idname = "EXPORT_OT_paper_model"
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
	def poll(self, context):
		return context.active_object and context.active_object.type=="MESH"
	def execute(self, context):
		unfolder=Unfolder(context.active_object)
		try:
			unfolder.save(self.properties)
			return {"FINISHED"}
		except UnfoldError as error:
			self.report(type="ERROR_INVALID_INPUT", message=error.args[0])
			return {"CANCELLED"}
		except:
			raise
	def invoke(self, context, event):
		sce=context.scene
		scale_length=sce.unit_settings.scale_length
		self.properties.output_size_x=sce.unfolder_output_size_x/scale_length
		self.properties.output_size_y=sce.unfolder_output_size_y/scale_length
		self.properties.output_dpi=sce.unfolder_output_dpi
		self.properties.output_pure=sce.unfolder_output_pure
		self.properties.bake_selected_to_active=sce.render.bake_active
		self.properties.sticker_width=0.005/scale_length
		wm = context.manager
		wm.add_fileselect(self)
		return {'RUNNING_MODAL'}
	def draw(self, context):
		layout = self.layout
		col = layout.column(align=True)
		col.label(text="Page size:")
		col.prop(self.properties, "output_size_x")
		col.prop(self.properties, "output_size_y")
		layout.prop(self.properties, "output_dpi")
		layout.prop(self.properties, "output_pure")
		col = layout.column()
		col.active = not self.properties.output_pure
		col.prop(self.properties, "bake_selected_to_active")
		layout.label(text="Document settings:")
		layout.prop(self.properties, "sticker_width")

class VIEW3D_paper_model(bpy.types.Panel):
	"""Blender UI Panel definition for Unfolder"""
	bl_space_type = 'VIEW_3D'
	bl_region_type = 'TOOLS'
	bl_label = "Export Paper Model"
	bpy.types.Scene.FloatProperty(attr="unfolder_output_size_x", name="Page Size X", description="Page width", default=0.210, soft_min=0.105, soft_max=0.841, subtype="UNSIGNED", unit="LENGTH")
	bpy.types.Scene.FloatProperty(attr="unfolder_output_size_y", name="Page Size Y", description="Page height", default=0.297, soft_min=0.148, soft_max=1.189, subtype="UNSIGNED", unit="LENGTH")
	bpy.types.Scene.FloatProperty(attr="unfolder_output_dpi", name="Unfolder DPI", description="Output resolution in points per inch", default=90, min=1, soft_min=30, soft_max=600, subtype="UNSIGNED")
	bpy.types.Scene.BoolProperty(attr="unfolder_output_pure", name="Pure Net", description="Do not bake the bitmap", default=True)

	def poll(self, context):
		return (context.active_object and context.active_object.type == 'MESH')

	def draw(self, context):
		layout = self.layout
		layout.operator("MESH_OT_make_unfoldable")
		col = layout.column()
		sub = col.column(align=True)
		sub.label(text="Page size:")
		sub.prop(bpy.context.scene, "unfolder_output_size_x", text="Width")
		sub.prop(bpy.context.scene, "unfolder_output_size_y", text="Height")
		col.prop(bpy.context.scene, "unfolder_output_dpi", text="DPI")
		col.prop(bpy.context.scene, "unfolder_output_pure")
		sub = col.column()
		sub.active = not bpy.context.scene.unfolder_output_pure
		sub.prop(context.scene.render, "bake_active", text="Bake Selected to Active")
		col.operator("export.paper_model", text="Export Net...")

def menu_func(self, context):
	self.layout.operator("export.paper_model", text="Paper Model (.svg)")

def register():
	bpy.types.register(VIEW3D_paper_model)
	bpy.types.register(MESH_OT_make_unfoldable)
	bpy.types.register(EXPORT_OT_paper_model)
	bpy.types.INFO_MT_file_export.append(menu_func)

def unregister():
	bpy.types.unregister(VIEW3D_paper_model)
	bpy.types.unregister(MESH_OT_make_unfoldable)
	bpy.types.unregister(EXPORT_OT_paper_model)
	bpy.types.INFO_MT_file_export.remove(menu_func)

if __name__ == "__main__":
	register()