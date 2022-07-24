# -*- coding: utf-8 -*-
# This script is Free software. Please share and reuse.
# ♡2022 Adam Dominec <adominec@gmail.com>

# This script reloads your addon every time you save a file.
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
import importlib
import types
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Pair:
    module: types.ModuleType
    stamp: int
    def __repr__(self):
        return self.module.__name__
    __str__ = __repr__


def timestamp(filename):
    return Path(filename).stat().st_mtime


def touch(filename):
    Path(filename).touch()


watchers = list()


def watcher_factory(script_name, interval=1.0):
    modules = list()
    init = None
    for name, module in sys.modules.items():
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        if any(filename.startswith(f"{path}/addons/{script_name}") for path in bpy.utils.script_paths()):
            pair = Pair(module, timestamp(filename))
            modules.append(pair)
            if filename.endswith("__init__.py"):
                init = pair
    print("Watching", modules)
    def watch_script():
        touched = list()
        for pair in modules:
            now = timestamp(pair.module.__file__)
            if now != pair.stamp:
                pair.stamp = now
                touched.append(pair)
        if touched:
            if init:
                init.module.unregister()
            for pair in touched:
                print("Auto-reload", pair)
                pair.module = importlib.reload(pair.module)
            if init:
                if init not in touched:
                    init.module = importlib.reload(init.module)
                init.module.register()
        return interval
    return watch_script


def start():
    prefs = bpy.context.preferences.addons[__name__].preferences
    script_name = prefs.script_name
    if script_name:
        watcher = watcher_factory(script_name, interval=prefs.interval or 1.0)
        watchers.append(watcher)
        bpy.app.timers.register(watcher, first_interval=prefs.interval)


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
    interval: bpy.props.FloatProperty(
        name="Interval", description="Number of seconds between subsequent checks for updates",
        default=1, soft_min=0.1, soft_max=60, subtype="UNSIGNED", unit="TIME")


    def draw(self, context):
        self.layout.prop(self, "script_name")
        self.layout.prop(self, "interval")


def register():
    bpy.utils.register_class(WatchPreferences)
    # wait three seconds for all addons to be loaded
    bpy.app.timers.register(start, first_interval=3.0)


def unregister():
    stop()
    bpy.utils.unregister_class(WatchPreferences)
