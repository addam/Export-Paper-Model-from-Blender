# -*- coding: utf-8 -*-
# This script is Free software. Please share and reuse.
# ♡2010-2021 Adam Dominec <adominec@gmail.com>

bl_info = {
    "name": "Export Paper Model",
    "author": "Addam Dominec",
    "version": (1, 3),
    "blender": (2, 83, 0),
    "location": "File > Export > Paper Model",
    "warning": "",
    "description": "Export printable net of the active mesh",
    "doc_url": "{BLENDER_MANUAL_URL}/addons/import_export/paper_model.html",
    "category": "Import-Export",
}

if "bpy" in locals():
    from importlib import reload
    reload(operator)
    reload(unfolder)
    reload(svg)
    reload(pdf)
    del reload

import bpy
from . import operator


def factory_update_addon_category(cls, prop):
    def func(self, context):
        if hasattr(bpy.types, cls.__name__):
            bpy.utils.unregister_class(cls)
        cls.bl_category = self[prop]
        bpy.utils.register_class(cls)
    return func


class PaperAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = "io_export_paper_model"
    unfold_category: bpy.props.StringProperty(
        name="Unfold Panel Category", description="Category in 3D View Toolbox where the Unfold panel is displayed",
        default="Paper", update=factory_update_addon_category(operator.VIEW3D_PT_paper_model_tools, 'unfold_category'))
    export_category: bpy.props.StringProperty(
        name="Export Panel Category", description="Category in 3D View Toolbox where the Export panel is displayed",
        default="Paper", update=factory_update_addon_category(operator.VIEW3D_PT_paper_model_settings, 'export_category'))

    def draw(self, context):
        sub = self.layout.column(align=True)
        sub.use_property_split = True
        sub.label(text="3D View Panel Category:")
        sub.prop(self, "unfold_category", text="Unfold Panel:")
        sub.prop(self, "export_category", text="Export Panel:")


module_classes = (
    operator.Unfold,
    operator.ExportPaperModel,
    operator.ClearAllSeams,
    operator.SelectIsland,
    operator.FaceList,
    operator.IslandList,
    operator.PaperModelSettings,
    operator.DATA_PT_paper_model_islands,
    operator.VIEW3D_PT_paper_model_tools,
    operator.VIEW3D_PT_paper_model_settings,
    PaperAddonPreferences,
)


def register():
    for cls in module_classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.paper_model = bpy.props.PointerProperty(
        name="Paper Model", description="Settings of the Export Paper Model script",
        type=operator.PaperModelSettings, options={'SKIP_SAVE'})
    bpy.types.Mesh.paper_island_list = bpy.props.CollectionProperty(
        name="Island List", type=operator.IslandList)
    bpy.types.Mesh.paper_island_index = bpy.props.IntProperty(
        name="Island List Index",
        default=-1, min=-1, max=100, options={'SKIP_SAVE'}, update=operator.island_index_changed)
    bpy.types.TOPBAR_MT_file_export.append(operator.menu_func_export)
    bpy.types.VIEW3D_MT_edit_mesh.prepend(operator.menu_func_unfold)
    # Force an update on the panel category properties
    prefs = bpy.context.preferences.addons[__name__].preferences
    prefs.unfold_category = prefs.unfold_category
    prefs.export_category = prefs.export_category


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(operator.menu_func_export)
    bpy.types.VIEW3D_MT_edit_mesh.remove(operator.menu_func_unfold)
    for cls in reversed(module_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()

