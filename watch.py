# -*- coding: utf-8 -*-
# This script is Free software. Please share and reuse.
# ♡2022 Adam Dominec <adominec@gmail.com>

# This script clicks "System -> Reload Scripts" for you every time you save your code.
# It starts when this addon is enabled, so once you specify the script name to be watched,
# your script will be watched for updates even after opening a new blend file.

bl_info = {
    "name": "Watch Reload",
    "author": "Addam Dominec",
    "version": (0, 1),
    "blender": (3, 0, 0),
    "location": "",
    "warning": "",
    "description": "Watch for script update and reload automatically",
    "doc_url": "",
    "category": "Development",
}


import bpy
import sys
import os


def timestamp(filename):
    return os.stat(filename).st_mtime


watchers = list()


def watcher_factory(script_name, interval=1.0):
    modules = dict()
    for name, module in sys.modules.items():
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        if any(filename.startswith(f"{path}/addons/{script_name}") for path in bpy.utils.script_paths()):
            modules[module] = timestamp(filename)
    print("Watching", modules)
    def watch_script():
        touched = False
        for module, stamp in modules.items():
            now = timestamp(module.__file__)
            if now != stamp:
                modules[module] = now
                touched = True
        if touched:
            bpy.ops.script.reload()
        return interval
    return watch_script


def start():
    prefs = bpy.context.preferences.addons[__name__].preferences
    script_name = prefs.script_name
    if script_name:
        watcher = watcher_factory(script_name)
        watchers.append(watcher)
        bpy.app.timers.register(watcher, first_interval=1.0, persistent=True)


def stop():
    for fn in watchers:
        try:
            bpy.app.timers.unregister(fn)
        except ValueError:
            pass


def restart(self, context):
    stop()
    start()


class WatchPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__
    script_name: bpy.props.StringProperty(
        name="Script Name", description="name of the script to be watched",
        default="io_export_paper_model", update=restart)

    def draw(self, context):
        self.layout.prop(self, "script_name")


def register():
    bpy.utils.register_class(WatchPreferences)
    start()


def unregister():
    stop()
    bpy.utils.unregister_class(WatchPreferences)
