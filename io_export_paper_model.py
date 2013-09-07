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

#### TODO:
# change UI 'Model Scale' to be a divisor, not a coefficient
# split islands bigger than selected page size
# UI elements to set line thickness and page size conveniently
# sanitize the constructors so that they don't edit their parent object
# apply island rotation and position before exporting, to simplify things
# s/verts/vertices/g
# SVG object doesn't need a 'pure_net' argument in constructor
# maybe Island would do with a list of points as well, set of vertices makes things more complicated
# why does UVVertex copy its position in constructor?

bl_info = {
	"name": "Export Paper Model",
	"author": "Addam Dominec",
	"version": (0, 8),
	"blender": (2, 6, 8),
	"api": 58966,
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
import bpy, bgl
import mathutils as M
try:
	from math import pi
except ImportError:
	pi = 3.141592653589783
try:
	from blist import blist
except ImportError:
	blist = list

priority_effect={
	'convex':0.5,
	'concave':1,
	'length':-0.05}
highlight_faces = list()

strf="{:.3f}".format

def sign(a):
	"""Return -1 for negative numbers, 1 for positive and 0 for zero."""
	return -1 if a < 0 else 1 if a > 0 else 0

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
	"""Generate consecutive pairs throughout the given sequence; at last, it gives elements last, first."""
	i=iter(sequence)
	previous=first=next(i)
	for this in i:
		yield previous, this
		previous=this
	yield this, first

def argmax_pair(array, key):
	l = len(array)
	mi, mj, m = None, None, None
	for i in range(l):
		for j in range(i+1, l):
			k = key(array[i], array[j])
			if not m or k > m:
				mi, mj, m = i, j, k
	return mi, mj

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

def create_blank_image(image_name, dimensions, alpha=1):
	"""BPy API doesn't allow to directly create a transparent image; this hack uses the New Image operator for it"""
	image_name = image_name[:20]
	obstacle = bpy.data.images.get(image_name)
	if obstacle:
		obstacle.name = image_name[0:-1] #when we create the new image, we want it to have *exactly* the name we assign
	bpy.ops.image.new(name=image_name, width=int(dimensions.x), height=int(dimensions.y), color=(1,1,1,alpha))
	image = bpy.data.images.get(image_name) #this time it is our new image
	if not image:
		print ("papermodel ERROR: could not get image", image_name)
	image.file_format = 'PNG'
	return image

class UnfoldError(ValueError):
	pass

class Unfolder:
	def __init__(self, ob):
		self.ob=ob
		self.mesh=Mesh(ob.data, ob.matrix_world)
		self.tex = None

	def prepare(self, create_uvmap=False, mark_seams=False):
		"""Something that should be part of the constructor - TODO """
		self.mesh.generate_cuts()
		self.mesh.finalize_islands()
		if create_uvmap:
			self.tex = self.mesh.save_uv()
		if mark_seams:
			self.mesh.mark_cuts()

	def save(self, properties):
		"""Export the document."""
		# Note about scale: input is direcly in blender length. finalize_islands multiplies everything by scale/page_size.y, SVG object multiplies everything by page_size.y*ppm.
		filepath=properties.filepath
		if filepath[-4:]==".svg" or filepath[-4:]==".png":
			filepath=filepath[0:-4]
		page_size = M.Vector((properties.output_size_x, properties.output_size_y)) # page size in meters
		printable_size = page_size - 2*properties.output_margin*M.Vector((1, 1)) # printable area size in meters
		print(page_size, printable_size)
		scale = bpy.context.scene.unit_settings.scale_length * properties.model_scale
		ppm = properties.output_dpi * 100 / 2.54 # pixels per meter
		self.mesh.mark_hidden_cuts((1e-3 if not properties.do_create_stickers else 0.5 * properties.style.outer_width) / (ppm * scale))
		if properties.do_create_numbers and properties.do_create_stickers:
			self.mesh.enumerate_islands()
		if properties.do_create_stickers:
			self.mesh.generate_stickers(properties.sticker_width * printable_size.y / scale, properties.do_create_numbers)
		elif properties.do_create_numbers:
			self.mesh.generate_numbers_alone(properties.sticker_width * printable_size.y / scale)
		text_height = 12/(printable_size.y*ppm) if properties.do_create_numbers else 0
		self.mesh.finalize_islands(scale_factor=scale / printable_size.y, space_at_bottom=text_height) # Scale everything so that page height is 1
		self.mesh.fit_islands(aspect_ratio = printable_size.x / printable_size.y)
		if not properties.output_pure:
			use_separate_images = properties.image_packing in ('ISLAND_LINK', 'ISLAND_EMBED')
			tex = self.mesh.save_uv(aspect_ratio=printable_size.x/printable_size.y, separate_image=use_separate_images, tex=self.tex)
			if not tex:
				raise UnfoldError("The mesh has no UV Map slots left. Either delete an UV Map or export pure net only.")
			rd = bpy.context.scene.render
			recall_selected_to_active, rd.use_bake_selected_to_active = rd.use_bake_selected_to_active, properties.bake_selected_to_active
			if properties.image_packing == 'PAGE_LINK':
				self.mesh.save_image(tex, filepath, printable_size * ppm)
			elif properties.image_packing == 'ISLAND_LINK':
				self.mesh.save_separate_images(tex, printable_size.y * ppm, filepath)
			elif properties.image_packing == 'ISLAND_EMBED':
				self.mesh.save_separate_images(tex, printable_size.y * ppm, do_embed=True)
			#revoke settings
			bpy.context.scene.render.use_bake_selected_to_active = recall_selected_to_active
			if not properties.do_create_uvmap:
				tex.active = True
				bpy.ops.mesh.uv_texture_remove()
		svg = SVG(page_size * ppm, printable_size.y*ppm, properties.style, properties.output_pure)
		svg.do_create_stickers = properties.do_create_stickers
		svg.margin = properties.output_margin*ppm
		svg.write(self.mesh, filepath)

class Mesh:
	"""Wrapper for Bpy Mesh"""
	
	def __init__(self, mesh, matrix):
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
			edge = self.edges[index]
			edge.choose_main_faces()
			if edge.main_faces:
				edge.calculate_angle()
	
	def generate_cuts(self):
		"""Cut the mesh so that it will be unfoldable."""
		
		twisted_faces = [face for face in self.faces.values() if face.is_twisted()]
		if twisted_faces:
			print ("There are {} twisted face(s) with ids: {}".format(len(twisted_faces), ", ".join(str(face.index) for face in twisted_faces)))
		
		self.islands = {Island(face) for face in self.faces.values()}
		# check for edges that are cut permanently
		edges = [edge for edge in self.edges.values() if not edge.force_cut and len(edge.faces) > 1]
		
		if edges:
			average_length = sum(edge.length for edge in edges) / len(edges)
			for edge in edges:
				edge.generate_priority(average_length)
			edges.sort(key = lambda edge:edge.priority, reverse=False)
			for edge in edges:
				if edge.length == 0:
					continue
				face_a, face_b = edge.main_faces
				island_a, island_b = face_a.uvface.island, face_b.uvface.island
				if len(island_b.faces) > len(island_a.faces):
					island_a, island_b = island_b, island_a
				if island_a is not island_b:
					if island_a.join(island_b, edge):
						self.islands.remove(island_b)
			
		for edge in self.edges.values():
			# some edges did not know until now whether their angle is convex or concave
			if edge.main_faces and edge.main_faces[0].uvface.flipped != edge.main_faces[1].uvface.flipped:
				edge.calculate_angle()
			# ensure that the order of faces corresponds to the order of uvedges
			if len(edge.uvedges) >= 2:
				reordered = [None, None]
				for uvedge in edge.uvedges:
					try:
						index = edge.main_faces.index(uvedge.uvface.face)
						reordered[index] = uvedge
					except ValueError:
						reordered.append(uvedge)
				edge.uvedges = reordered

		# construct a linked list from each island's boundary
		for island in self.islands:
			neighbor_lookup = {(uvedge.va if uvedge.uvface.flipped else uvedge.vb): uvedge for uvedge in island.boundary_sorted}
			for uvedge in island.boundary_sorted:
				uvedge.neighbor_right = neighbor_lookup[uvedge.vb if uvedge.uvface.flipped else uvedge.va]
				uvedge.neighbor_right.neighbor_left = uvedge
		
		return True
	
	def mark_hidden_cuts(self, distance):
		epsilon = distance**2
		for edge in self.edges.values():
			# mark edges of flat polygons that need not be drawn, and also cuts whose uvedges are very close
			if edge.is_main_cut and len(edge.uvedges) >= 2 and edge.uvedges[0].is_similar(edge.uvedges[1], epsilon):
				edge.cut_is_hidden = True
	
	def mark_cuts(self):
		"""Mark cut edges in the original mesh so that the user can see"""
		for edge in self.edges.values():
			edge.data.use_seam = len(edge.uvedges) > 1 and edge.is_main_cut
	
	def generate_stickers(self, default_width, do_create_numbers=True):
		"""Add sticker faces where they are needed."""
		def uvedge_priority(uvedge):
			"""Retuns whether it is a good idea to stick something on this edge's face"""
			#TODO: it should take into account overlaps with faces and with other stickers
			return uvedge.uvface.face.area / sum((vb-va).length for (va, vb) in pairs(uvedge.uvface.verts))
		for edge in self.edges.values():
			if edge.is_main_cut and not edge.cut_is_hidden and len(edge.uvedges) >= 2:
				uvedge_a, uvedge_b = edge.uvedges[:2]
				if uvedge_priority(uvedge_a) < uvedge_priority(uvedge_b):
					uvedge_a, uvedge_b = uvedge_b, uvedge_a
				target_island = uvedge_a.island
				left_edge, right_edge = uvedge_a.neighbor_left.edge, uvedge_a.neighbor_right.edge
				if do_create_numbers:
					for uvedge in [uvedge_b] + edge.uvedges[2:]:
						if (uvedge.neighbor_left.edge is not right_edge or uvedge.neighbor_right.edge is not left_edge) and\
								uvedge not in (uvedge_a.neighbor_left, uvedge_a.neighbor_right):
							# it is perhaps not easy to see that these uvedges should be sticked together. So, create an arrow and put the index on all stickers
							target_island.sticker_numbering += 1
							index = str(target_island.sticker_numbering)
							if {'6','9'} < set(index) < {'6','8','9','0'}:
								# if index consists of the digits 6, 8, 9, 0 only and contains 6 or 9, make it distinguishable
								index += "."
							target_island.add_marker(Arrow(uvedge_a, default_width, index))
							break
					else:
						# if all uvedges to be sticked are easy to see, create no numbers
						index = None
				else:
					index = None
				uvedge_b.island.add_marker(Sticker(uvedge_b, default_width, index, target_island))
			elif len(edge.uvedges) > 2:
				index = None
				target_island = edge.uvedges[0].island
			if len(edge.uvedges) > 2:
				for uvedge in edge.uvedges[2:]:
					uvedge.island.add_marker(Sticker(uvedge, default_width, index, target_island))
	
	def generate_numbers_alone(self, size):
		global_numbering = 0
		for edge in self.edges.values():
			if edge.is_main_cut and not edge.cut_is_hidden and len(edge.uvedges) >= 2:
				global_numbering += 1
				index = str(global_numbering)
				if ('6' in index or '9' in index) and set(index) <= {'6','8','9','0'}:
					# if index consists of the digits 6, 8, 9, 0 only and contains 6 or 9, make it distinguishable
					index += "."
				for uvedge in edge.uvedges:
					uvedge.island.add_marker(NumberAlone(uvedge, index, size))
	
	def enumerate_islands(self):
		for num, island in enumerate(self.islands, 1):
			island.number = num
			island.generate_label()
	
	def finalize_islands(self, scale_factor=1, space_at_bottom=0, do_enumerate=False):
		for island in self.islands:
			island.apply_scale(scale_factor)
			island.generate_bounding_box(space_at_bottom=space_at_bottom)

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
	
	def save_uv(self, aspect_ratio=1, separate_image=False, tex=None): #page_size is in pixels
		bpy.ops.object.mode_set()
		#note: expecting that the active object's data is self.mesh
		if not tex:
			tex = self.data.uv_textures.new()
			if not tex:
				return None
		tex.name = "Unfolded"
		tex.active = True
		loop = self.data.uv_layers[self.data.uv_layers.active_index] # TODO: this is somehow dirty, but I don't see a nicer way in the API
		if separate_image:
			for island in self.islands:
				island.save_uv_separate(loop)
		else:
			for island in self.islands:
				island.save_uv(loop, aspect_ratio)
		return tex
	
	def save_image(self, tex, filename, page_size_pixels:M.Vector):
		rd = bpy.context.scene.render
		recall_margin, rd.bake_margin = rd.bake_margin, 0
		recall_clear, rd.use_bake_clear = rd.use_bake_clear, False

		tex.active = True
		loop = self.data.uv_layers[self.data.uv_layers.active_index]
		aspect_ratio = page_size_pixels.x / page_size_pixels.y
		for page in self.pages:
			#image=bpy.data.images.new(name="Unfolded "+self.data.name+" "+page.name, width=int(page_size.x), height=int(page_size.y))
			image = create_blank_image("{} {} Unfolded".format(self.data.name[:14], page.name), page_size_pixels, alpha=1)
			image.filepath_raw = page.image_path = "{}_{}.png".format(filename, page.name)
			texfaces=tex.data
			for island in page.islands:
				for uvface in island.faces:
					texfaces[uvface.face.index].image=image
			bpy.ops.object.bake_image()
			image.save()
			for island in page.islands:
				for uvface in island.faces:
					texfaces[uvface.face.index].image=None
			image.user_clear()
			bpy.data.images.remove(image)
		rd.bake_margin=recall_margin
		rd.use_bake_clear=recall_clear
	
	def save_separate_images(self, tex, scale, filepath=None, do_embed=False):
		if do_embed:
			try:
				from base64 import encodebytes as b64encode
				from os import remove
			except ImportError:
				raise UnfoldError("Embedding images is not supported on your system")
		else:
			try:
				from os import mkdir
				from os.path import dirname, basename
				imagedir = "{path}/{directory}".format(path=dirname(filepath), directory = basename(filepath))
				mkdir(imagedir)
			except ImportError:
				raise UnfoldError("This method of image packing is not supported by your system.")
			except OSError:
				pass #imagedir already existed
		rd=bpy.context.scene.render
		recall_margin=rd.bake_margin; rd.bake_margin=0
		recall_clear=rd.use_bake_clear; rd.use_bake_clear=False
		
		texfaces=tex.data
		for i, island in enumerate(self.islands, 1):
			image_name = "unfolder_temp_{}".format(id(island)%100) if do_embed else "{} isl{}".format(self.data.name[:15], i)
			image = create_blank_image(image_name, island.bounding_box * scale, alpha=0)
			image.filepath_raw = image_path = "{}.png".format(image_name) if do_embed else "{}/island{}.png".format(imagedir, i)
			for uvface in island.faces:
				texfaces[uvface.face.index].image=image
			bpy.ops.object.bake_image()
			image.save()
			for uvface in island.faces:
				texfaces[uvface.face.index].image = None
			image.user_clear()
			bpy.data.images.remove(image)
			
			if do_embed:
				with open(image_path, 'rb') as imgf:
					island.embedded_image = b64encode(imgf.read()).decode('ascii')
				remove(image_path)
			else:
				island.image_path = image_path
				
		
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
		
class Edge:
	"""Wrapper for BPy Edge"""
	
	def __init__(self, edge, mesh, matrix=1):
		self.data=edge
		self.va=mesh.verts[edge.vertices[0]]	
		self.vb=mesh.verts[edge.vertices[1]]
		self.vect=self.vb.co-self.va.co
		self.length=self.vect.length
		self.faces=list()
		self.uvedges=list() # important: if self.main_faces is set, then self.uvedges[:2] must be the ones corresponding to self.main_faces.
		                    # It is assured at the time of finishing mesh.generate_cuts
		
		self.force_cut = bool(edge.use_seam) # such edges will always be cut
		self.main_faces = None # two faces that can be connected in the island
		self.is_main_cut = True # defines whether the two main faces are connected; all the others will be automatically treated as cut
		self.cut_is_hidden = False # for cuts inside flat faces that need not actually be drawn as cut
		self.priority=None
		self.angle = None
		self.va.edges.append(self)
		self.vb.edges.append(self)
	
	def choose_main_faces(self):
		"""Choose two main faces that might get connected in an island"""
		if len(self.faces) == 2:
			self.main_faces = self.faces
		elif len(self.faces) > 2:
			# find (with brute force) the pair of indices whose faces have the most similar normals
			i, j = argmax_pair(self.faces, key=lambda a, b: a.normal.dot(b.normal))
			self.main_faces = self.faces[i], self.faces[j]		
	
	def calculate_angle(self):
		"""Calculate the angle between the main faces"""
		face_a, face_b = self.main_faces
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
	
	def is_cut(self, face=None):
		"""Optional argument 'face' defines who is asking (useful for edges with more than two faces connected)"""
		#Return whether there is a cut between the two main faces
		if face is None or (self.main_faces and face in self.main_faces):
			return self.is_main_cut
		#All other faces (third and more) are automatically treated as cut
		else:
			return True
	
	def other_uvedge(self, this):
		"""Get an uvedge of this edge that is not the given one - or None if no other uvedge was found."""
		if len(self.uvedges) < 2:
			return None
		return self.uvedges[1] if this is self.uvedges[0] else self.uvedges[0]

class Face:
	"""Wrapper for BPy Face"""
	def __init__(self, bpy_face, mesh, matrix=1):
		self.data = bpy_face 
		self.index = bpy_face.index
		self.edges = list()
		self.verts = [mesh.verts[i] for i in bpy_face.vertices]
		self.loop_start = bpy_face.loop_start
		self.area = bpy_face.area
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
	def is_twisted(self):
		if len(self.verts) > 3:
			center = vectavg(self.verts)
			plane_d = center.dot(self.normal)
			diameter = max((center-vertex.co).length for vertex in self.verts)
			for vertex in self.verts:
				# check coplanarity
				if abs(vertex.co.dot(self.normal) - plane_d) > diameter*0.01: #TODO: this threshold should be editable or well chosen at least
					return True
		return False
	def __hash__(self):
		return hash(self.index)

class Island:
	def __init__(self, face=None):
		"""Create an Island from a single Face"""
		self.faces=list()
		self.edges=set()
		self.verts=set()
		self.fake_verts = list()
		self.pos=M.Vector((0,0))
		self.offset=M.Vector((0,0))
		self.angle=0
		self.is_placed=False
		self.bounding_box=M.Vector((0,0))

		self.image_path = None
		self.embedded_image = None
		
		if face:
			self.add(UVFace(face, self))
		
		self.boundary_sorted = list(self.edges)
		self.boundary_sorted.sort()
		
		self.scale = 1
		self.markers = list()
		self.sticker_numbering = 0
		self.label = None

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
		self.faces.append(uvface)
	
	def add_marker(self, marker):
		self.fake_verts.extend(marker.bounds)
		self.markers.append(marker)
	
	def convex_hull(self) -> list:
		"""Returns a list of Vectors that forms the best fitting convex polygon."""
		def make_convex_curve(points, cross=cross_product):
			"""Remove points from given vert list so that the result poly is a convex curve (works for both top and bottom)."""
			result = list()
			for point in points:
				while len(result) >= 2 and cross((point - result[-1]), (result[-1] - result[-2])) >= 0:
					result.pop()
				result.append(point)
			return result
		points = list(self.fake_verts)
		points.extend(vertex.co for vertex in self.verts)
		points.sort(key=lambda point: point.x)
		points_top = make_convex_curve(points)
		points_bottom = make_convex_curve(reversed(points))
		#remove left and right ends and concatenate the lists to form a polygon in the right order
		return points_top[:-1] + points_bottom[:-1]
	
	def generate_bounding_box(self, space_at_bottom=0):
		"""Find the rotation for the optimal bounding box and calculate its dimensions."""
		def bounding_box_score(size):
			"""Calculate the score - the bigger result, the better box."""
			return 1/(size.x*size.y)
		points_convex = self.convex_hull()
		if not points_convex:
			raise UnfoldError("Error, check topology of the mesh object (failed to calculate a convex hull)")
		#go through all edges and search for the best solution
		best_score = 0
		best_box = (0, M.Vector((0,0)), M.Vector((0,0))) #(angle, box, offset) for the best score
		for point_a, point_b in pairs(points_convex):
			if point_a == point_b:
				continue
			angle = angle2d(point_b - point_a)
			rot = M.Matrix.Rotation(angle, 2)
			#find the dimensions in both directions
			rotated = [rot*point for point in points_convex]
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
		box.y += space_at_bottom
		offset.y -= space_at_bottom
		self.angle = angle
		self.bounding_box = box
		self.offset = -offset
	
	def apply_scale(self, scale=1):
		if scale != 1:
			self.scale *= scale
			for vertex in self.verts:
				vertex.co *= scale
			for point in self.fake_verts:
				point *= scale
	
	def generate_label(self, label=None, abbreviation=None):
		abbr = abbreviation or str(self.number)
		if not set('69NZMWpbqd').isdisjoint(abbr) and set('6890oOxXNZMWIlpbqd').issuperset(abbr):
			abbr += "."
		self.label = label or ("{}:Island {}".format(abbreviation, self.number) if abbreviation else "Island: {}".format(self.number))
		self.abbreviation = abbr
	
	def save_uv(self, tex, aspect_ratio=1):
		"""Save UV Coordinates of all UVFaces to a given UV texture
		tex: UV Texture layer to use (BPy MeshUVLoopLayer struct)
		page_size: size of the page in pixels (vector)"""
		texface = tex.data
		for uvface in self.faces:
			rot = M.Matrix.Rotation(self.angle, 2)
			for i, uvvertex in enumerate(uvface.verts):
				uv = rot * uvvertex.co + self.offset + self.pos
				texface[uvface.face.loop_start + i].uv[0] = uv.x / aspect_ratio
				texface[uvface.face.loop_start + i].uv[1] = uv.y
	
	def save_uv_separate(self, tex):
		"""Save UV Coordinates of all UVFaces to a given UV texture, spanning from 0 to 1
		tex: UV Texture layer to use (BPy MeshUVLoopLayer struct)
		page_size: size of the page in pixels (vector)"""
		texface = tex.data
		scale_x, scale_y = 1/self.bounding_box.x, 1/self.bounding_box.y
		for uvface in self.faces:
			rot = M.Matrix.Rotation(self.angle, 2)
			for i, uvvertex in enumerate(uvface.verts):
				uv = rot * uvvertex.co + self.offset
				texface[uvface.face.loop_start + i].uv[0] = uv.x * scale_x
				texface[uvface.face.loop_start + i].uv[1] = uv.y * scale_y

class Page:
	"""Container for several Islands"""
	def __init__(self, num=1):
		self.islands=list()
		self.name="page"+str(num)
		self.image_path = None
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
			return hash(self.co.x) ^ hash(self.co.y) # this is dirty: hash of such UVVertex can change
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
		self.uvface = uvface
		if edge:
			self.edge = edge
			edge.uvedges.append(self)
		#Every UVEdge is attached to only one UVFace. UVEdges are doubled as needed, because they both have to point clockwise around their faces
	def update(self):
		"""Update data if UVVertices have moved"""
		self.min, self.max = (self.va, self.vb) if (self.va < self.vb) else (self.vb, self.va)
		y1, y2 = self.va.co.y, self.vb.co.y
		self.bottom, self.top = (y1, y2) if y1 < y2 else (y2, y1)
	def is_similar(self, other, epsilon = 1e-6):
		if self.island is not other.island:
			return False
		pair_a, pair_b = (other.va, other.vb) if self.uvface.flipped ^ other.uvface.flipped else (other.vb, other.va)
		return (self.va - pair_a).length_squared < epsilon and (self.vb - pair_b).length_squared < epsilon
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
	def __init__(self, face:Face, island:Island):
		"""Creace an UVFace from a Face and a fixed edge.
		face: Face to take coordinates from
		island: Island to register itself in
		fixed_edge: Edge to connect to (that already has UV coordinates)"""
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
		self.edges=list()
		edge_by_verts=dict()
		for edge in face.edges:
			edge_by_verts[(edge.va.index, edge.vb.index)]=edge
			edge_by_verts[(edge.vb.index, edge.va.index)]=edge
		for va, vb in pairs(self.verts):
			uvedge = UVEdge(va, vb, island, self, edge_by_verts[(va.vertex.index, vb.vertex.index)])
			self.edges.append(uvedge)
			island.edges.add(uvedge)
	
class Marker:
	"""Various graphical elements linked to the net, but not being parts of the mesh"""
	def __init__(self):
		self.bounds = list()

class Arrow(Marker):
	def __init__(self, uvedge, size, index):
		self.text = str(index)
		edge = (uvedge.vb - uvedge.va) if not uvedge.uvface.flipped else (uvedge.va - uvedge.vb)
		self.center = (uvedge.va.co + uvedge.vb.co) / 2
		self.size = size
		sin, cos = edge.y/edge.length, edge.x/edge.length
		self.rot = M.Matrix(((cos, -sin), (sin, cos)))
		tangent = edge.normalized()
		normal = M.Vector((tangent.y, -tangent.x))
		self.bounds = [self.center, self.center + (1.2*normal+tangent)*size, self.center + (1.2*normal-tangent)*size]

class Sticker(Marker):
	"""Sticker face"""
	def __init__(self, uvedge, default_width=0.005, index=None, target_island=None):
		"""Sticker is directly attached to the given UVEdge"""
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
			cos_a = min(max(cos_a, (edge*other_edge)/(edge.length**2)), 1) #angles between pi/3 and 0; fix for math errors
			sin_a = (1-cos_a**2)**0.5
			len_b = min(len_a, (edge.length*sin_a)/(sin_a*cos_b+sin_b*cos_a))
			len_a = 0 if sin_a == 0 else min(sticker_width/sin_a, (edge.length-len_b*cos_b)/cos_a)
		elif second_vertex == other_first:
			cos_b = min(max(cos_b, (edge*other_edge)/(edge.length**2)), 1) #angles between pi/3 and 0; fix for math errors
			sin_b = (1-cos_b**2)**0.5
			len_a = min(len_a, (edge.length*sin_b)/(sin_a*cos_b+sin_b*cos_a))
			len_b = 0 if sin_b == 0 else min(sticker_width/sin_b, (edge.length-len_a*cos_a)/cos_b)
		v3 = UVVertex(second_vertex.co + M.Matrix(((cos_b, -sin_b), (sin_b, cos_b))) * edge * len_b/edge.length)
		v4 = UVVertex(first_vertex.co + M.Matrix(((-cos_a, -sin_a), (sin_a, -cos_a))) * edge * len_a/edge.length)
		if v3.co != v4.co:
			self.vertices=[second_vertex, v3, v4, first_vertex]
		else:
			self.vertices=[second_vertex, v3, first_vertex]
		
		sin, cos = edge.y/edge.length, edge.x/edge.length
		self.rot = M.Matrix(((cos, -sin), (sin, cos)))
		self.width = sticker_width * 0.9
		self.text = "{}:{}".format(target_island.label, index) if index and target_island is not uvedge.island else index or None
		self.center = (uvedge.va.co + uvedge.vb.co) / 2 + self.rot*M.Vector((0, self.width*0.2))
		self.bounds = [v3.co, v4.co, self.center] if v3.co != v4.co else [v3.co, self.center]

class NumberAlone(Marker):
	"""Numbering inside the island describing edges to be sticked"""
	def __init__(self, uvedge, index, default_size=0.005):
		"""Sticker is directly attached to the given UVEdge"""
		edge = (uvedge.va - uvedge.vb) if not uvedge.uvface.flipped else (uvedge.vb - uvedge.va)

		self.size = default_size# min(default_size, edge.length/2)
		sin, cos = edge.y/edge.length, edge.x/edge.length
		self.rot = M.Matrix(((cos, -sin), (sin, cos)))
		self.text = index
		self.center = (uvedge.va.co + uvedge.vb.co) / 2 - self.rot*M.Vector((0, self.size*1.2))
		self.bounds = [self.center]

class SVG:
	"""Simple SVG exporter"""
	def __init__(self, page_size_pixels:M.Vector, scale, style, pure_net=True):
		"""Initialize document settings.
		page_size_pixels: document dimensions in pixels
		pure_net: if True, do not use image"""
		self.page_size = page_size_pixels
		self.scale = scale
		self.pure_net = pure_net
		self.style = style
		self.margin = 0
	def format_vertex(self, vector, rot=1, pos=M.Vector((0,0))):
		"""Return a string with both coordinates of the given vertex."""
		vector = rot*vector + pos
		return str((vector.x)*self.scale + self.margin) + " " + str((1-vector.y)*self.scale+self.margin)
	def write(self, mesh, filename):
		"""Write data to a file given by its name."""
		line_through = " L ".join #utility function
		format_color = lambda vec: "#{:02x}{:02x}{:02x}".format(round(vec[0]*255), round(vec[1]*255), round(vec[2])*255)
		format_style = {'SOLID':"none", 'DOT':"0.2,4", 'DASH':"4,8", 'LONGDASH':"6,3", 'DASHDOT':"8,4,2,4"}
		rows = "\n".join
		for num, page in enumerate(mesh.pages):
			with open(filename+"_"+page.name+".svg", 'w') as f:
				f.write("<?xml version='1.0' encoding='UTF-8' standalone='no'?>\n")
				f.write("<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' version='1.1' x='0px' y='0px' width='" + str(self.page_size.x) + "px' height='" + str(self.page_size.y) + "px'>")
				f.write("""<style type="text/css">
					path {{fill:none; stroke-width:{outer_width:.2}px; stroke-linecap:square; stroke-linejoin:bevel; stroke-dasharray:none}}
					path.outer {{stroke:{outer_color}; stroke-dasharray:{outer_style}; stroke-dashoffset:0; stroke-width:{outer_width:.2}px; stroke-opacity: {outer_alpha:.2}}}
					path.convex {{stroke:{convex_color}; stroke-dasharray:{convex_style}; stroke-dashoffset:0; stroke-width:{convex_width:.2}px; stroke-opacity: {convex_alpha:.2}}}
					path.concave {{stroke:{concave_color}; stroke-dasharray:{concave_style}; stroke-dashoffset:0; stroke-width:{concave_width:.2}px; stroke-opacity: {concave_alpha:.2}}}
					path.background {{stroke:#fff}}
					path.outer_background {{stroke:#fff; stroke-width:{outline:.2}px}}
					path.sticker {{fill: {sticker_fill}; stroke: {sticker_color}; fill-opacity: {sticker_alpha:.2}; stroke-width:{sticker_width:.2}; stroke-opacity: 1}}
					path.arrow {{fill: #000;}}
					text {{font-size: 12px; font-style: normal; fill: {text_color}; fill-opacity: {text_alpha:.2}; stroke: none;}}
					text.scaled {{font-size: 1px;}}
					tspan {{text-anchor:middle;}}
				</style>""".format(outer_color=format_color(self.style.outer_color), outer_alpha=self.style.outer_color[3], outer_style=format_style[self.style.outer_style],
					convex_color=format_color(self.style.convex_color), convex_alpha=self.style.convex_color[3], convex_style=format_style[self.style.convex_style],
					concave_color=format_color(self.style.concave_color), concave_alpha=self.style.concave_color[3], concave_style=format_style[self.style.concave_style],
					sticker_fill=format_color(self.style.sticker_fill), sticker_color=format_color(self.style.sticker_color), sticker_alpha=self.style.sticker_fill[3],
					text_color=format_color(self.style.text_color), text_alpha=self.style.text_color[3],
					outer_width=self.style.outer_width, convex_width=self.style.convex_width, concave_width=self.style.concave_width,
					sticker_width=self.style.sticker_width, outline=1.5*self.style.outer_width))
				if page.image_path:
					f.write("<image transform='matrix(1 0 0 1 0 0)' width='{}' height='{}' xlink:href='file://{}'/>\n".format(self.page_size.x, self.page_size.y, page.image_path))
				if len(page.islands) > 1:
					f.write("<g>")
				for island in page.islands:
					f.write("<g>")
					if island.image_path:
						f.write("<image transform='translate({pos})' width='{width}' height='{height}' xlink:href='file://{path}'/>\n".format(
							pos=self.format_vertex(island.pos + M.Vector((0, island.bounding_box.y))), width=island.bounding_box.x*self.scale, height=island.bounding_box.y*self.scale,
							path=island.image_path))
					elif island.embedded_image:
						f.write("<image transform='translate({pos})' width='{width}' height='{height}' xlink:href='data:image/png;base64,".format(
							pos=self.format_vertex(island.pos + M.Vector((0, island.bounding_box.y))), width=island.bounding_box.x*self.scale, height=island.bounding_box.y*self.scale,
							path=island.image_path))
						f.write(island.embedded_image)
						f.write("'/>\n")
					rot = M.Matrix.Rotation(island.angle, 2)
					pos = island.pos + island.offset
					
					data_outer, data_convex, data_concave = list(), list(), list()
					for uvedge in island.edges:
						edge = uvedge.edge
						data_uvedge = "M " + line_through((self.format_vertex(vertex.co, rot, pos) for vertex in (uvedge.va, uvedge.vb)))
						if not edge.is_cut(uvedge.uvface.face) or edge.cut_is_hidden:
							if uvedge.uvface.flipped ^ (uvedge.va.vertex.index > uvedge.vb.vertex.index): # each uvedge is in two opposite-oriented variants; we want to add each only once
								if edge.angle > 0.01:
									data_convex.append(data_uvedge)
								elif edge.angle < -0.01:
									data_concave.append(data_uvedge)
						else:
							data_outer.append(data_uvedge)
					if data_outer:
						if not self.pure_net:
							f.write("<path class='outer_background' d='" + rows(data_outer) + "'/>")
						f.write("<path class='outer' d='" + rows(data_outer) + "'/>")
					if not self.pure_net and (data_convex or data_concave):
						f.write("<path class='background' d='" + rows(data_convex + data_concave) + "'/>")
					if data_convex: f.write("<path class='convex' d='" + rows(data_convex) + "'/>")
					if data_concave: f.write("<path class='concave' d='" + rows(data_concave) + "'/>")
					
					if island.label:
						island_label = "^Island: {}^".format(island.label) if island.bounding_box.x*self.scale > 80 else island.label # just a guess of the text width
						f.write("<text transform='translate({x} {y})'><tspan>{label}</tspan></text>".format(
							x=self.scale * (island.bounding_box.x*0.5 + island.pos.x) + self.margin, y=self.scale * (1 - island.pos.y) + self.margin,
							label=island_label))
					data_markers = list()
					format_matrix = lambda mat: " ".join(" ".join(map(str, col)) for col in mat)
					for marker in island.markers:
						if type(marker) is Sticker:
							if self.do_create_stickers:
								text = "<text class='scaled' transform='matrix({mat} {pos})'><tspan>{index}</tspan></text>".format(
									index=marker.text,
									pos=self.format_vertex(marker.center, rot, pos),
									mat=format_matrix(marker.width * island.scale * self.scale * rot * marker.rot)) if marker.text else ""
								data_markers.append("<g><path class='sticker' d='M {data} Z'/>{text}</g>".format(
									data=line_through((self.format_vertex(vertex.co, rot, pos) for vertex in marker.vertices)),
									text=text))
							elif marker.text:
								data_markers.append("<text class='scaled' transform='matrix({mat} {pos})'><tspan>{index}</tspan></text>".format(
									index=marker.text,
									pos=self.format_vertex(marker.center, rot, pos),
									mat=format_matrix(marker.width * island.scale * self.scale * rot * marker.rot)))
						elif type(marker) is Arrow:
							size = marker.size * island.scale * self.scale
							data_markers.append("<g><path transform='matrix({mat} {arrow_pos})' class='arrow' d='M 0 0 L 1 1 L 0 0.25 L -1 1 Z'/><text class='scaled' transform='matrix({scale} 0 0 {scale} {pos})'><tspan>{index}</tspan></text></g>".format(
								index=marker.text,
								arrow_pos=self.format_vertex(marker.center, rot, pos),
								scale=size,
								pos=self.format_vertex(marker.center + marker.rot*marker.size*island.scale*M.Vector((0, -0.9)), rot, pos - marker.size*island.scale*M.Vector((0, 0.4))),
								mat=format_matrix(size * rot * marker.rot)))
						elif type(marker) is NumberAlone:
							size = marker.size * island.scale * self.scale
							data_markers.append("<text class='scaled' transform='matrix({mat} {pos})'><tspan>{index}</tspan></text>".format(
								index=marker.text,
								pos=self.format_vertex(marker.center, rot, pos),
								mat=format_matrix(size * rot * marker.rot)))
					if data_markers:
						f.write("<g>" + rows(data_markers) + "</g>") #Stickers are separate paths in one group
					f.write("</g>")
				
				if len(page.islands) > 1:
					f.write("</g>")
				f.write("</svg>")

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
	do_create_uvmap = bpy.props.BoolProperty(name="Create UVMap", description="Create a new UV Map showing the islands and page layout", default=False)
	unfolder = None
	
	@classmethod
	def poll(cls, context):
		return context.active_object and context.active_object.type=="MESH"
		
	def draw(self, context):
		layout = self.layout
		col = layout.column()
		col.active = not self.unfolder or len(self.unfolder.mesh.data.uv_textures) < 8
		col.prop(self.properties, "do_create_uvmap")
		layout.label(text="Edge Cutting Factors:")
		col = layout.column(align=True)
		col.label(text="Face Angle:")
		col.prop(self.properties, "priority_effect_convex", text="Convex")
		col.prop(self.properties, "priority_effect_concave", text="Concave")
		layout.prop(self.properties, "priority_effect_length", text="Edge Length")
	
	def execute(self, context):
		global priority_effect
		sce = bpy.context.scene
		priority_effect['convex']=self.priority_effect_convex
		priority_effect['concave']=self.priority_effect_concave
		priority_effect['length']=self.priority_effect_length
		recall_mode = context.object.mode
		bpy.ops.object.mode_set(mode='OBJECT')
		recall_display_islands, sce.io_paper_model_display_islands = sce.io_paper_model_display_islands, False
		

		self.unfolder = unfolder = Unfolder(context.active_object)
		unfolder.prepare(mark_seams=True, create_uvmap=self.do_create_uvmap)

		island_list = context.scene.island_list
		island_list.clear() #remove previously defined islands
		for island in unfolder.mesh.islands:
			#add islands to UI list and set default descriptions
			list_item = island_list.add()
			list_item.name = "{} ({} faces)".format(island.label, len(island.faces))
			list_item.island = island
			#add faces' IDs to the island
			for uvface in island.faces:
				face_list_item = list_item.faces.add()
				face_list_item.id = uvface.face.index
		sce.island_list_index = -1
		list_selection_changed(sce, bpy.context)

		unfolder.mesh.data.show_edge_seams = True
		bpy.ops.object.mode_set(mode=recall_mode)
		sce.io_paper_model_display_islands = recall_display_islands
		return {'FINISHED'}

def page_size_preset_changed(self, context):
	"""Update the actual document size to correct values"""
	if self.page_size_preset == 'A4':
		self.output_size_x = 0.210
		self.output_size_y = 0.297
	elif self.page_size_preset == 'A3':
		self.output_size_x = 0.297
		self.output_size_y = 0.420
	elif self.page_size_preset == 'US_LETTER':
		self.output_size_x = 0.216
		self.output_size_y = 0.279
	elif self.page_size_preset == 'US_LEGAL':
		self.output_size_x = 0.216
		self.output_size_y = 0.356

class PaperModelStyle(bpy.types.PropertyGroup):
	line_styles = [
			('SOLID', "Solid (----)", "Solid line"),
			('DOT', "Dots (. . .)", "Dotted line"),
			('DASH', "Short Dashes (- - -)", "Solid line"),
			('LONGDASH', "Long Dashes (-- --)", "Solid line"),
			('DASHDOT', "Dash-dotted (-- .)", "Solid line")]
	outer_color = bpy.props.FloatVectorProperty(name="Outer Lines", description="Color of net outline", default=(0.0, 0.0, 0.0, 1.0), min=0, max=1, subtype='COLOR', size=4)
	outer_style = bpy.props.EnumProperty(name="Outer Lines Drawing Style", description="Drawing style of net outline", default='SOLID', items=line_styles)
	outer_width = bpy.props.FloatProperty(name="Outer Lines Thickness", description="Thickness of net outline, in pixels", default=1.5, min=0, soft_max=10, precision=1)
	convex_color = bpy.props.FloatVectorProperty(name="Inner Convex Lines", description="Color of lines to be folded to a convex angle", default=(0.0, 0.0, 0.0, 1.0), min=0, max=1, subtype='COLOR', size=4)
	convex_style = bpy.props.EnumProperty(name="Convex Lines Drawing Style", description="Drawing style of lines to be folded to a convex angle", default='DASH', items=line_styles)
	convex_width = bpy.props.FloatProperty(name="Convex Lines Thickness", description="Thickness of concave lines, in pixels", default=1, min=0, soft_max=10, precision=1)
	concave_color = bpy.props.FloatVectorProperty(name="Inner Concave Lines", description="Color of lines to be folded to a concave angle", default=(0.0, 0.0, 0.0, 1.0), min=0, max=1, subtype='COLOR', size=4)
	concave_style = bpy.props.EnumProperty(name="Concave Lines Drawing Style", description="Drawing style of lines to be folded to a concave angle", default='DASHDOT', items=line_styles)
	concave_width = bpy.props.FloatProperty(name="Concave Lines Thickness", description="Thickness of concave lines, in pixels", default=1, min=0, soft_max=10, precision=1)
	sticker_fill = bpy.props.FloatVectorProperty(name="Tabs Fill", description="Fill color of sticking tabs", default=(1.0, 1.0, 1.0, 0.4), min=0, max=1, subtype='COLOR', size=4)
	sticker_color = bpy.props.FloatVectorProperty(name="Tabs Outline", description="Outline color of sticking tabs", default=(0.0, 0.0, 0.0), min=0, max=1, subtype='COLOR', size=3)
	sticker_width = bpy.props.FloatProperty(name="Tabs Outline Thickness", description="Thickness of tabs outer line, in pixels", default=1, min=0, soft_max=10, precision=1)
	text_color = bpy.props.FloatVectorProperty(name="Text Color", description="Color of all text used in the document", default=(0.0, 0.0, 0.0, 1.0), min=0, max=1, subtype='COLOR', size=4)
bpy.utils.register_class(PaperModelStyle)

class ExportPaperModel(bpy.types.Operator):
	"""Blender Operator: save the selected object's net and optionally bake its texture"""
	bl_idname = "export_mesh.paper_model"
	bl_label = "Export Paper Model"
	bl_description = "Export the selected object's net and optionally bake its texture"
	filepath = bpy.props.StringProperty(name="File Path", description="Target file to save the SVG")
	filename = bpy.props.StringProperty(name="File Name", description="Name of the file")
	directory = bpy.props.StringProperty(name="Directory", description="Directory of the file")
	page_size_preset = bpy.props.EnumProperty(name="Page Size", description="Size of the exported document", default='A4', update=page_size_preset_changed, items=[
			('USER', "User defined", "User defined paper size"),
			('A4', "A4", "International standard paper size"),
			('A3', "A3", "International standard paper size"),
			('US_LETTER', "Letter", "North American paper size"),
			('US_LEGAL', "Legal", "North American paper size")])
	output_size_x = bpy.props.FloatProperty(name="Page Width", description="Width of the exported document", default=0.210, soft_min=0.105, soft_max=0.841, subtype="UNSIGNED", unit="LENGTH")
	output_size_y = bpy.props.FloatProperty(name="Page Height", description="Height of the exported document", default=0.297, soft_min=0.148, soft_max=1.189, subtype="UNSIGNED", unit="LENGTH")
	output_margin = bpy.props.FloatProperty(name="Page Margin", description="Distance from page borders to the printable area", default=0.005, min=0, soft_max=0.1, subtype="UNSIGNED", unit="LENGTH")
	output_dpi = bpy.props.FloatProperty(name="Unfolder DPI", description="Resolution of images and lines in pixels per inch", default=90, min=1, soft_min=30, soft_max=600, subtype="UNSIGNED")
	output_pure = bpy.props.BoolProperty(name="Pure Net", description="Do not bake the bitmap", default=True)
	bake_selected_to_active = bpy.props.BoolProperty(name="Selected to Active", description="Bake selected to active (if not exporting pure net)", default=True)
	do_create_stickers = bpy.props.BoolProperty(name="Create Tabs", description="Create gluing tabs around the net (useful for paper)", default=True)
	do_create_numbers = bpy.props.BoolProperty(name="Create Numbers", description="Enumerate edges to make it clear which edges should be sticked together", default=True)
	sticker_width = bpy.props.FloatProperty(name="Tabs and Text Size", description="Width of gluing tabs and their numbers", default=0.005, soft_min=0, soft_max=0.05, subtype="UNSIGNED", unit="LENGTH")
	image_packing = bpy.props.EnumProperty(name="Image Packing Method", description="Method of attaching baked image(s) to the SVG", default='PAGE_LINK', items=[
			('PAGE_LINK', "Single Linked", "Bake one image per page of output"),
			('ISLAND_LINK', "Linked", "Bake images separately for each island and save them in a directory"),
			('ISLAND_EMBED', "Embedded", "Bake images separately for each island and embed them into the SVG")])
	model_scale = bpy.props.FloatProperty(name="Scale", description="Coefficient of all dimensions when exporting", default=1, soft_min=0.0001, soft_max=1.0, subtype="FACTOR")
	do_create_uvmap = bpy.props.BoolProperty(name="Create UVMap", description="Create a new UV Map showing the islands and page layout", default=False)
	ui_expanded_document = bpy.props.BoolProperty(name="Show Document Settings Expanded", description="Shows the box 'Document Settings' expanded in user interface", default=True)
	ui_expanded_style = bpy.props.BoolProperty(name="Show Style Settings Expanded", description="Shows the box 'Colors and Style' expanded in user interface", default=False)
	style = bpy.props.PointerProperty(type=PaperModelStyle)
	
	unfolder=None
	largest_island_ratio=0
	
	@classmethod
	def poll(cls, context):
		return context.active_object and context.active_object.type=='MESH'
	
	def execute(self, context):
		try:
			self.unfolder.save(self.properties)
			self.report({'INFO'}, "Saved {}-page document".format(len(self.unfolder.mesh.pages)))
			return {'FINISHED'}
		except UnfoldError as error:
			self.report(type={'ERROR_INVALID_INPUT'}, message=error.args[0])
			return {'CANCELLED'}
		except:
			raise
	def get_scale_ratio(self, sce):
		if min(self.output_size_x, self.output_size_y) <= 2*self.output_margin:
			return False
		ratio = self.unfolder.mesh.largest_island_ratio(M.Vector((self.output_size_x-2*self.output_margin, self.output_size_y-2*self.output_margin)))
		return ratio * self.model_scale * sce.unit_settings.scale_length
	def invoke(self, context, event):
		sce=context.scene
		self.bake_selected_to_active = sce.render.use_bake_selected_to_active
		
		self.object = context.active_object
		self.unfolder = Unfolder(self.object)
		self.unfolder.prepare(create_uvmap=self.do_create_uvmap)
		scale_ratio = self.get_scale_ratio(sce)
		if scale_ratio > 1:
			self.model_scale = 0.95/scale_ratio
		wm = context.window_manager
		wm.fileselect_add(self)
		return {'RUNNING_MODAL'}
	
	def draw(self, context):
		layout = self.layout
		layout.label(text="Model scale:")
		layout.prop(self.properties, "model_scale")
		scale_ratio = self.get_scale_ratio(context.scene)
		if scale_ratio > 1:
			layout.label(text="An island is "+strf(scale_ratio)+"x bigger than page", icon="ERROR")
		elif scale_ratio > 0:
			layout.label(text="Largest island is 1/"+strf(1/scale_ratio)+" of page")
		layout.prop(self.properties, "do_create_uvmap")

		box = layout.box()
		row = box.row(align=True)
		row.prop(self.properties, "ui_expanded_document", text="", icon=('TRIA_DOWN' if self.ui_expanded_document else 'TRIA_RIGHT'), emboss=False)
		row.label(text="Document Settings")
		
		if self.ui_expanded_document:
			box.prop(self.properties, "page_size_preset")
			col = box.column(align=True)
			col.active = self.page_size_preset == 'USER'
			col.prop(self.properties, "output_size_x")
			col.prop(self.properties, "output_size_y")
			box.prop(self.properties, "output_margin")
			box.prop(self.properties, "output_dpi")
			col = box.column()
			col.prop(self.properties, "do_create_stickers")
			col.prop(self.properties, "do_create_numbers")
			col = box.column()
			col.active = self.do_create_stickers or self.do_create_numbers
			col.prop(self.properties, "sticker_width")
			
			box.prop(self.properties, "output_pure")
			col = box.column()
			if len(self.object.data.uv_textures) == 8:
				col.label(text="No UV slots left, pure net is the only option.", icon="ERROR")
			col.active = not self.output_pure
			col.prop(self.properties, "bake_selected_to_active", text="Bake Selected to Active")
			col.prop(self.properties, "image_packing", text="Images")
		
		box = layout.box()
		row = box.row(align=True)
		row.prop(self.properties, "ui_expanded_style", text="", icon=('TRIA_DOWN' if self.ui_expanded_style else 'TRIA_RIGHT'), emboss=False)
		row.label(text="Colors and Style")
		
		if self.ui_expanded_style:
			col = box.column()
			col.prop(self.style, "outer_color")
			col.prop(self.style, "outer_width", text="Width (pixels)")
			col.prop(self.style, "outer_style", text="Style")
			col = box.column()
			col.prop(self.style, "convex_color")
			col.prop(self.style, "convex_width", text="Width (pixels)")
			col.prop(self.style, "convex_style", text="Style")
			col = box.column()
			col.prop(self.style, "concave_color")
			col.prop(self.style, "concave_width", text="Width (pixels)")
			col.prop(self.style, "concave_style", text="Style")
			col = box.column()
			col.active = self.do_create_stickers
			col.prop(self.style, "sticker_fill")
			col.prop(self.style, "sticker_color")
			col.prop(self.style, "sticker_width", text="Outline width (pixels)")
			box.prop(self.style, "text_color")

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
			box.template_list('UI_UL_list', 'io_paper_model_island_list', sce, 'island_list', sce, 'island_list_index', rows=1, maxrows=5)
			# The first one is the identifier of the registered UIList to use (if you want only the default list,
			# with no custom draw code, use "UI_UL_list").
			# layout.template_list("MATERIAL_UL_matslots_example", "", obj, "material_slots", obj, "active_material_index")
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
		
		layout.prop(sce, "io_paper_model_display_tabs", icon='RESTRICT_VIEW_OFF')
		layout.operator("export_mesh.paper_model")
	
def display_islands(self, context):
	#TODO: save the vertex positions and don't recalculate them always
	#TODO: don't use active object, but rather save the object itself
	if context.active_object != display_islands.object:
		return
	ob = context.active_object
	mesh = ob.data
	bgl.glMatrixMode(bgl.GL_PROJECTION)
	perspMatrix = context.space_data.region_3d.perspective_matrix
	perspBuff = bgl.Buffer(bgl.GL_FLOAT, (4,4), perspMatrix.transposed())
	bgl.glLoadMatrixf(perspBuff)
	bgl.glMatrixMode(bgl.GL_MODELVIEW)
	objectBuff = bgl.Buffer(bgl.GL_FLOAT, (4,4), ob.matrix_world.transposed())
	bgl.glLoadMatrixf(objectBuff)
	bgl.glEnable(bgl.GL_BLEND)
	bgl.glBlendFunc(bgl.GL_SRC_ALPHA, bgl.GL_ONE_MINUS_SRC_ALPHA);
	bgl.glEnable(bgl.GL_POLYGON_OFFSET_FILL)
	bgl.glPolygonOffset(0, -10) #offset in Zbuffer to remove flicker
	bgl.glPolygonMode(bgl.GL_FRONT_AND_BACK, bgl.GL_FILL)
	bgl.glColor4f(1.0, 0.4, 0.0, self.io_paper_model_islands_alpha)
	global highlight_faces
	for face_id in highlight_faces:
		face = mesh.polygons[face_id]
		bgl.glBegin(bgl.GL_POLYGON)
		for vertex_id in face.vertices:
			vertex = mesh.vertices[vertex_id]
			bgl.glVertex4f(*vertex.co.to_4d())
		bgl.glEnd()
	bgl.glPolygonOffset(0.0, 0.0)
	bgl.glDisable(bgl.GL_POLYGON_OFFSET_FILL)
	bgl.glLoadIdentity()
display_islands.handle = None
display_islands.object = None

def display_islands_changed(self, context):
	"""Switch highlighting islands on/off"""
	if self.io_paper_model_display_islands:
		if not display_islands.handle:
			display_islands.handle = bpy.types.SpaceView3D.draw_handler_add(display_islands, (self, context), 'WINDOW', 'POST_VIEW')
	else:
		if display_islands.handle:
			bpy.types.SpaceView3D.draw_handler_remove(display_islands.handle, 'WINDOW')
			display_islands.handle = None

def list_selection_changed(self, context):
	"""Update the island highlighted in 3D View"""
	global highlight_faces
	if self.island_list_index >= 0:
		list_item = self.island_list[self.island_list_index]
		highlight_faces = [face.id for face in list_item.faces]
		display_islands.object = context.active_object
	else:
		highlight_faces = list()
		display_islands.object = None

def label_changed(self, context):
	if len(self.abbreviation > 3):
		self.abbreviation = self.abbreviation[:3]
	self.island.generate_label(self.label, self.abbreviation)
	self.label = self.island.label
	self.abbreviation = self.island.abbreviation
	self.name = "{} ({} faces)".format(self.label, len(self.faces))

class FaceList(bpy.types.PropertyGroup):
	id = bpy.props.IntProperty(name="Face ID")
class IslandList(bpy.types.PropertyGroup):
	faces = bpy.props.CollectionProperty(type=FaceList, name="Faces", description="Faces belonging to this island")
	label = bpy.props.StringProperty(name="Label", description="Label on this island", default="", update=label_changed)
	abbreviation = bpy.props.StringProperty(name="Abbreviation", description="Three-letter label to use when there is not enough space", default="", update=label_changed)
bpy.utils.register_class(FaceList)
bpy.utils.register_class(IslandList)

#flip = bm.edges.layers.int.new("flip_tab")
#bmm.edges[0][bmm.edges.layers.int["flip_tab"]]

def display_tabs(self, context):
	if context.active_object.type != 'MESH':
		return
	from bmesh import new as BMesh
	bm = BMesh()
	ob = context.active_object
	bm.from_mesh(ob.data)
	
	bgl.glMatrixMode(bgl.GL_PROJECTION)
	perspMatrix = context.space_data.region_3d.perspective_matrix
	perspBuff = bgl.Buffer(bgl.GL_FLOAT, (4,4), perspMatrix.transposed())
	bgl.glLoadMatrixf(perspBuff)
	bgl.glMatrixMode(bgl.GL_MODELVIEW)
	objectBuff = bgl.Buffer(bgl.GL_FLOAT, (4,4), ob.matrix_world.transposed())
	bgl.glLoadMatrixf(objectBuff)
	bgl.glEnable(bgl.GL_POLYGON_OFFSET_LINE)
	bgl.glPolygonOffset(0, -10) #offset in Zbuffer to remove flicker
	polygonMode = bgl.Buffer(bgl.GL_INT, 2)
	bgl.glGetIntegerv(bgl.GL_POLYGON_MODE, polygonMode)
	bgl.glPolygonMode(bgl.GL_FRONT_AND_BACK, bgl.GL_LINE)
	bgl.glColor3f(1.0, 0.2, 0.0)
	
	linear_component = ob.matrix_world.to_3x3()
	
	for edge in bm.edges:
		if len(edge.link_faces) < 1:
			continue
		face = edge.link_faces[0] # use custom edge layer to pick the correct one

		shear = edge.verts[1].co - edge.verts[0].co
		offset = linear_component.inverted() * (linear_component*face.normal).cross(linear_component*shear)
		shear /= (linear_component*shear).length
		offset /= (linear_component*offset).length
		
		bgl.glBegin(bgl.GL_POLYGON)
		bgl.glVertex3f(*edge.verts[0].co)
		bgl.glVertex3f(*edge.verts[1].co)
		bgl.glVertex3f(*(edge.verts[1].co + offset))
		bgl.glVertex3f(*(edge.verts[0].co + offset))
		bgl.glEnd()

	bgl.glPolygonOffset(0.0, 0.0)
	bgl.glPolygonMode(bgl.GL_FRONT_AND_BACK, polygonMode[0])
	del polygonMode
	bgl.glDisable(bgl.GL_POLYGON_OFFSET_LINE)
	bgl.glLoadIdentity()
display_tabs.handle = None

def display_tabs_changed(self, context):
	if self.io_paper_model_display_tabs:
		if not display_tabs.handle:
			display_tabs.handle = bpy.types.SpaceView3D.draw_handler_add(display_tabs, (self, context), 'WINDOW', 'POST_VIEW')
	else:
		if display_tabs.handle:
			bpy.types.SpaceView3D.draw_handler_remove(display_tabs.handle, 'WINDOW')
			display_tabs.handle = None
	

def register():
	bpy.utils.register_module(__name__)

	bpy.types.Scene.io_paper_model_display_islands = bpy.props.BoolProperty(name="Highlight selected island", update=display_islands_changed)
	bpy.types.Scene.io_paper_model_islands_alpha = bpy.props.FloatProperty(name="Highlight Alpha", description="Alpha value for island highlighting", min=0.0, max=1.0, default=0.3)
	bpy.types.Scene.island_list = bpy.props.CollectionProperty(type=IslandList, name= "Island List", description= "")
	bpy.types.Scene.island_list_index = bpy.props.IntProperty(name="Island List Index", default= -1, min= -1, max= 100, update=list_selection_changed)
	bpy.types.Scene.io_paper_model_display_tabs = bpy.props.BoolProperty(name="Display sticking tabs", update=display_tabs_changed)
	bpy.types.Scene.io_paper_model_islands_alpha = bpy.props.FloatProperty(name="Highlight Alpha", description="Alpha value for island highlighting", min=0.0, max=1.0, default=0.3)
	bpy.types.INFO_MT_file_export.append(menu_func)

def unregister():
	bpy.utils.unregister_module(__name__)
	bpy.types.INFO_MT_file_export.remove(menu_func)
	if display_islands.handle:
		bpy.types.SpaceView3D.draw_handler_remove(display_islands.handle, 'WINDOW')
		display_islands.handle = None
	if display_tabs.handle:
		bpy.types.SpaceView3D.draw_handler_remove(display_tabs.handle, 'WINDOW')
		display_tabs.handle = None

if __name__ == "__main__":
	register()
