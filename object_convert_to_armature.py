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
import bmesh
from itertools import repeat
from math import asin, pi


def add_armature(mesh_object):
    armature = bpy.data.armatures.new(f"{mesh_object.name} Armature")
    armature_object = bpy.data.objects.new(armature.name, object_data=armature)
    armature_object.matrix_local = mesh_object.matrix_local
    bpy.context.scene.collection.objects.link(armature_object)
    
    modifier = mesh_object.modifiers.new("Folding Armature", 'ARMATURE')
    modifier.use_bone_envelopes = False
    modifier.use_vertex_groups = True
    modifier.object = armature_object
    return armature_object, armature


def add_bone(name, armature, head, tail):
    bo = armature.edit_bones.new(name)
    bo.head = head
    bo.tail = tail
    return bo


def prepare_bones(init_face):
    bones = list()
    loops = 0
    visited_faces = set()
    queue = [(None, init_face, None)]  # Edge, face, parent_face
    while queue:
        edge, face, parent_face = queue.pop()
        if face in visited_faces:
            loops += 1  # we arrived at the same face from two different directions
            continue
        visited_faces.add(face)
        tail = face.calc_center_median_weighted()
        if edge:
            a, b = [v.co for v in edge.verts]
            head = (a + b) / 2
            tail -= (tail - head).project(b - a)
        else:  # root bone
            head = tail.copy()
            head.z -= 1
        bones.append((face, parent_face, head, tail))
        for loop in face.loops:
            if loop.edge is not edge:
                queue.extend((loop.edge, l.face, face) for l in loop.link_loops)
    return bones, loops


def lock_pose_rotation(pose_bone):
    pose_bone.lock_rotation[0:3] = False, True, True
    pose_bone.lock_location[0:3] = True, True, True
    pose_bone.lock_scale[0:3] = True, True, True
    pose_bone.lock_ik_x, pose_bone.lock_ik_y, pose_bone.lock_ik_z = False, True, True
    pose_bone.rotation_mode = 'XYZ'

def main(context):
    mesh_object = context.object
    if context.mode == 'EDIT':
        bm = bmesh.from_edit_mesh(mesh_object.data)
    else:
        bm = bmesh.new()
        bm.from_mesh(mesh_object.data)
    bm.verts.ensure_lookup_table()
    
    armature_object, armature = add_armature(mesh_object)
    bones, loops = prepare_bones(bm.faces.active or next(iter(bm.faces)))
    context.view_layer.objects.active = armature_object
    recall_mode = context.mode
    # Create bones (requires edit mode)
    bpy.ops.object.mode_set(mode='EDIT')
    face_bone = dict()
    for face, parent_face, head, tail in bones:
        group = mesh_object.vertex_groups.new(name=f"Face {face.index}")  # Create a vertex group for this face
        group.add([v.index for v in face.verts], 1, 'ADD')
        bone = add_bone(group.name, armature, head, tail)
        bone.align_roll(face.normal)
        bone.parent = face_bone.get(parent_face)
        face_bone[face] = bone
    bone_face = {eb.name: f for f, eb in face_bone.items()}
    # Set transform locks (requires object mode)
    bpy.ops.object.mode_set(mode='OBJECT')
    for pose_bone in armature_object.pose.bones:
        bone = pose_bone.bone
        parent = bone.parent
        if parent is None:
            pose_bone.lock_scale[0:3] = True, True, True
        else:
            lock_pose_rotation(pose_bone)
            face = bone_face[bone.name]
            parent_face = bone_face[parent.name]
            # note: this almost works (and it does not need the bone_face dict)
            #  face_normal = bone.matrix_local.col[2].to_3d()
            #  parent_normal = parent.matrix_local.col[2].to_3d()
            #  angle = face_normal.angle(parent_normal)
            angle = face.normal.angle(parent_face.normal)
            pose_bone.rotation_euler.x = angle
    context.view_layer.objects.active = mesh_object
    bpy.ops.object.mode_set(mode=recall_mode)
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
