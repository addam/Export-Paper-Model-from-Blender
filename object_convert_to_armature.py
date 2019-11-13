# coding: utf-8
# This script is Free software. Please share and reuse.
# ♡2010-2019 Adam Dominec <adominec@gmail.com>


bl_info = {
    "name": "Convert Mesh to Armature",
    "author": "Addam Dominec",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "Object > Convert to Armature",
    "warning": "",
    "description": "Generate an armature with a single bone controlling each face",
    "category": "Object",
    "wiki_url": "http://wiki.blender.org/index.php/Extensions:2.6/Py/Scripts/Object/Convert_to_Armature",
}

import bpy
from itertools import repeat
from functools import reduce
from mathutils import Vector
from math import asin, pi


faces_by_edge=dict();
edge_by_verts=dict();


def pairs(sequence):
    return zip(sequence, sequence[1:]+sequence[:1])


def get_edge(verts):
    global edge_by_verts
    va, vb = verts
    if va > vb:
        va, vb = vb, va
    if (va, vb) not in edge_by_verts:
        print("Verts %i %i not in list" % (va, vb))
    return edge_by_verts.get((va, vb), None)


def get_edges(face, exclude=None):
    edges = map(get_edge, pairs(face.vertices))
    return filter(lambda edge: edge and edge is not exclude, edges)


def get_faces(edge, exclude=None):
    global faces_by_edge
    if exclude:
        return [face for face in faces_by_edge.get(edge, ()) if face != exclude]
    else:
        return faces_by_edge.get(edge, [])


def vertex_avg(mesh, indices):
    return reduce(Vector.__add__, [mesh.vertices[i].co for i in indices]) / len(indices)


def add_bone(name, armature, head, tail):
    bo = armature.edit_bones.new(name)
    bo.head = head
    bo.tail = tail
    return bo


def main(context):
    global faces_by_edge
    global edge_by_verts
    faces_by_edge.clear()
    edge_by_verts.clear()
    visited_faces = set()
    loops = 0
    
    bpy.ops.object.mode_set()
    mesh_object = context.active_object
    mesh = mesh_object.data
    
    #Create a nicer structure above the mesh data
    for edge in mesh.edges:
        if not edge.use_seam:
            va, vb = edge.vertices
            if va > vb:
                va, vb = vb, va
            edge_by_verts[(va, vb)] = edge
    for face in mesh.polygons:
        for edge in get_edges(face):
            if edge not in faces_by_edge:
                faces_by_edge[edge]=list()
            faces_by_edge[edge].append(face)
    
    #Create an armature
    armature = bpy.data.armatures.new(f"{mesh_object.name} Armature")
    armature_object = bpy.data.objects.new(armature.name, object_data=armature)
    context.scene.collection.objects.link(armature_object)
    armature_object.matrix_local = mesh_object.matrix_local
    
    modifier = mesh_object.modifiers.new("Folding Armature", 'ARMATURE')
    modifier.use_bone_envelopes = False
    modifier.use_vertex_groups = True
    modifier.object = armature_object
    
    #Generate the bones
    context.view_layer.objects.active = armature_object
    bpy.ops.object.mode_set(mode='EDIT')
    angles = dict() #bone -> folding angle
    queue = [(None, mesh.polygons[mesh.polygons.active], None, None)] #Edge, face and a bone connecting these two and parent in the tree
    while queue:
        edge, face, parent_bone, parent_face = queue.pop()
        if face in visited_faces:
            loops += 1 #We went to the same face from two different directions
            continue
        visited_faces.add(face)
        vgroup = mesh_object.vertex_groups.new(name=f"Face {face.index}") #Create a vertex group for this face
        vgroup.add(face.vertices, 1, 'ADD')
        if edge:
            edge_vector = mesh.vertices[edge.vertices[0]].co - mesh.vertices[edge.vertices[1]].co
            head = vertex_avg(mesh, edge.vertices)
            tail = vertex_avg(mesh, face.vertices)
            tail -= (tail-head).project(edge_vector)
        else: #root bone
            tail = vertex_avg(mesh, face.vertices)
            head = tail.copy()
            head.z -= 1
        bone = add_bone(vgroup.name, armature, head, tail)
        bone.align_roll(face.normal)
        bone.parent = parent_bone
        if edge: #all except the root bone
            depth = bone.vector.dot(parent_face.normal)/parent_face.normal.length
            clamped = min(1, max(-1, depth/bone.vector.length))
            try:
                angle = asin(clamped)
            except ValueError:
                print("Depth: {}, |vector|: {}".format(depth, bone.vector.length))
                angle = 0
            if face.normal.dot(parent_face.normal) < 0:
                angle = pi - angle
            angles[bone.name] = angle
        for next_edge in get_edges(face, exclude=edge):
            queue += zip(repeat(next_edge), get_faces(next_edge, exclude=face), repeat(bone), repeat(face))
    bpy.ops.object.mode_set()
    #Set transform locks (must be done in object mode)
    pose_bone = armature_object.pose.bones[0]
    pose_bone.lock_scale[0:3] = True, True, True
    for pose_bone in armature_object.pose.bones[1:]: 
        pose_bone.lock_rotation[0:3] = False, True, True
        pose_bone.lock_location[0:3] = True, True, True
        pose_bone.lock_scale[0:3] = True, True, True
        pose_bone.lock_ik_x, pose_bone.lock_ik_y, pose_bone.lock_ik_z = False, True, True
        pose_bone.rotation_mode = 'XYZ'
        pose_bone.rotation_euler.x = -angles[pose_bone.name]
    context.view_layer.objects.active = mesh_object
    if loops:
        raise ValueError(loops)

class OBJECT_OT_convert_to_armature(bpy.types.Operator):
    '''Generate an armature with a single bone controlling each face. The mesh must be a tree-like structure for this to make sense. Active face is used for main bone.'''
    bl_idname = "object.convert_to_armature"
    bl_label = "Convert to Armature"
    bl_description = "Generate an armature from the active mesh"

    @classmethod
    def poll(cls, context):
        return context.active_object.type == 'MESH'

    def execute(self, context):
        try:
            main(context)
        except ValueError as E:
            if isinstance(E.args[0], int):
                self.report({'ERROR', 'ERROR_INVALID_INPUT'}, "There is a loop of connected faces. Use Export Paper Model add-on and EdgeSplit modifier to eliminate them.\n" "Otherwise, the armature may be unusable.")
            else:
                raise
        return {'FINISHED'}


def menu_func(self, context):
    self.layout.operator(OBJECT_OT_convert_to_armature.bl_idname, text="Convert to Armature")


def register():
    bpy.utils.register_class(OBJECT_OT_convert_to_armature)
    bpy.types.VIEW3D_MT_object.append(menu_func)


def unregister():
    bpy.types.VIEW3D_MT_object.remove(menu_func)
    bpy.utils.unregister_class(OBJECT_OT_convert_to_armature)


if __name__ == "__main__":
    register()
