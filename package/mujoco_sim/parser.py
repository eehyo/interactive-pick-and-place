"""
MuJoCo parser and lightweight viewer utilities used across the workspace.
"""
from __future__ import annotations

import os
import time
import mujoco
import glfw
import pathlib
import cv2
import warnings
import numpy as np
try:
    from termcolor import colored
except Exception:
    def colored(text, color=None):
        return text

from typing import Optional
from threading import Lock

# MuJoCo version
MUJOCO_VERSION = tuple(map(int,mujoco.__version__.split('.')))

from package.utility.utils import ( # type: ignore
    get_idxs,
    get_monitor_size,
    xy2xyz,
    TicToc,
)
from package.utility.viz import ( # type: ignore
    get_colors,
    meters2xyz,
)
from package.kinematics.transforms import ( # type: ignore
    np_uv,
    t2p,
    t2r,
    pr2t,
    rpy2r,
    get_R_from_twopoints,
    r2quat,
)
from package.mujoco_sim.mujoco_utils import (
    normalize_mujoco_joint_types,
)

def _print_with_text_color(text,text_color=None):
    """
    Print text with an optional short color code.

    Parameters:
        text (str): Text to print.
        text_color (str | None): Optional color code such as "r", "g", "b", or "c".
    """
    color_map = {
        "r":"red",
        "g":"green",
        "b":"blue",
        "c":"cyan",
    }
    if text_color is None:
        print(text)
        return

    color_key = str(text_color).lower()
    if color_key in color_map:
        print(colored(text,color_map[color_key]))
    else:
        print(text)

# MinimalCallbacks class
class MinimalCallbacks:
    def __init__(self):
        """
        Minimal GLFW callbacks for mouse and keyboard interaction.
        """
        # Mouse
        self._gui_lock                   = Lock()
        self._button_left_pressed        = False
        self._button_right_pressed       = False
        self._left_double_click_pressed  = False
        self._right_double_click_pressed = False
        self._last_left_click_time       = None
        self._last_right_click_time      = None
        self._last_mouse_x               = 0
        self._last_mouse_y               = 0
        self._paused                     = False
        self._render_every_frame         = True
        self._time_per_render            = 1/60.0
        self._run_speed                  = 1.0
        self._loop_count                 = 0
        self._advance_by_one_step        = False
        
        # Keyboard 
        self._key_pressed                = None
        self._is_key_pressed             = False
        
        # Keyboard buffer
        self._key_pressed_set            = set()
        self._key_repeated_set           = set()
    
    def _key_callback(self, window, key, scancode, action, mods):
        """
        GLFW key callback to handle key press and release events.
        """
        # Flags for key pressed 
        is_key_pressed  = (action==glfw.PRESS)
        is_key_released = (action==glfw.RELEASE)
        is_key_repeated = (action==glfw.REPEAT)

        # Add and discard keys
        if is_key_pressed:
            self._key_pressed_set.add(key)
        if is_key_repeated:
            self._key_repeated_set.add(key)
        if is_key_released:
            # Remove from pressed and repeated lists (if present)
            self._key_pressed_set.discard(key)
            self._key_repeated_set.discard(key)

        # Quit (escape)
        if (key==glfw.KEY_ESCAPE):
            glfw.set_window_should_close(window, True)

        # Return
        return
    
    def _cursor_pos_callback(self, window, xpos, ypos):
        """ 
        GLFW cursor position callback to handle mouse movement events.
        """
        # Only proceed if a mouse button is pressed
        if not (self._button_left_pressed or self._button_right_pressed):
            return
        
        # Determine action based on mouse buttons and modifier keys
        mod_shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
            glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        if self._button_right_pressed:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._button_left_pressed:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        # Calculate mouse movement deltas
        dx = int(self._scale * xpos) - self._last_mouse_x
        dy = int(self._scale * ypos) - self._last_mouse_y
        width, height = glfw.get_framebuffer_size(window)

        # Update last mouse positions
        with self._gui_lock:
            if self.pert.active:
                mujoco.mjv_movePerturb(
                    self.model,
                    self.data,
                    action,
                    dx / height,
                    dy / height,
                    self.scn,
                    self.pert)
            else:
                mujoco.mjv_moveCamera(
                    self.model,
                    action,
                    dx / height,
                    dy / height,
                    self.scn,
                    self.cam)
                
        # Update last mouse positions
        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)

    def _mouse_button_callback(self, window, button, act, mods):
        """
        GLFW callback for mouse button actions.

        Handles button presses, releases, and double-click detection.
        Also activates or deactivates perturbation of objects in the scene.
        """        
        self._button_left_pressed = button == glfw.MOUSE_BUTTON_LEFT and act == glfw.PRESS
        self._button_right_pressed = button == glfw.MOUSE_BUTTON_RIGHT and act == glfw.PRESS

        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)

        # detect a left- or right- doubleclick
        self._left_double_click_pressed = False
        self._right_double_click_pressed = False
        time_now = glfw.get_time()

        if self._button_left_pressed:
            if self._last_left_click_time is None:
                self._last_left_click_time = glfw.get_time()

            time_diff = (time_now - self._last_left_click_time)
            if time_diff > 0.01 and time_diff < 0.3:
                self._left_double_click_pressed = True
            self._last_left_click_time = time_now

        if self._button_right_pressed:
            if self._last_right_click_time is None:
                self._last_right_click_time = glfw.get_time()

            time_diff = (time_now - self._last_right_click_time)
            if time_diff > 0.01 and time_diff < 0.3:
                self._right_double_click_pressed = True
            self._last_right_click_time = time_now

        # set perturbation
        key = mods == glfw.MOD_CONTROL
        newperturb = 0
        if key and self.pert.select > 0:
            # right: translate, left: rotate
            if self._button_right_pressed:
                newperturb = mujoco.mjtPertBit.mjPERT_TRANSLATE
            if self._button_left_pressed:
                newperturb = mujoco.mjtPertBit.mjPERT_ROTATE

            # perturbation onste: reset reference
            if newperturb and not self.pert.active:
                mujoco.mjv_initPerturb(
                    self.model, self.data, self.scn, self.pert)
        self.pert.active = newperturb
        # 3D release
        if act == glfw.RELEASE:
            self.pert.active = 0

    def _scroll_callback(self, window, x_offset, y_offset):
        """
        GLFW callback for mouse wheel scrolling.

        Zooms the camera in/out based on scroll direction.
        """        
        with self._gui_lock:
            mujoco.mjv_moveCamera(
                self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * y_offset, self.scn, self.cam)            

# MuJoCoMinimalViewer class
class MuJoCoMinimalViewer(MinimalCallbacks):
    def __init__(
            self,
            model,
            data,
            mode              = 'window',
            title             = "MuJoCo Minimal Viewer",
            width             = None,
            height            = None,
            maxgeom           = 50000,
            perturbation      = True,
            x_offset          = None,
            y_offset          = None,
        ):
        super().__init__()

        # Store parameters
        self.model       = model
        self.data        = data
        self.render_mode = mode
        self.is_alive    = True # viewer is alive
        self.CONFIG_PATH = pathlib.Path.joinpath(pathlib.Path.home(), ".config/mujoco_viewer/config.yaml")
        
        # glfw init
        glfw.init()
        
        # Get monitor size
        monitor_w,monitor_h = glfw.get_video_mode(glfw.get_primary_monitor()).size
        if not width:
            width = monitor_w
        if not height:
            height = monitor_h
        if self.render_mode == 'offscreen':
            glfw.window_hint(glfw.VISIBLE, 0)

        # Create window
        self.maxgeom = maxgeom
        self.window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # Get framebuffer size
        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self.window)

        # install callbacks only for 'window' mode
        if self.render_mode == 'window':
            window_width, _ = glfw.get_window_size(self.window)
            # Mouse callbacks use framebuffer pixels, so keep the conversion
            # from logical window coordinates for HiDPI displays.
            self._scale = framebuffer_width * 1.0 / window_width
            # set callbacks
            glfw.set_cursor_pos_callback(
                self.window, self._cursor_pos_callback)
            glfw.set_mouse_button_callback(
                self.window, self._mouse_button_callback)
            glfw.set_scroll_callback(self.window, self._scroll_callback)
            glfw.set_key_callback(self.window, self._key_callback)

        # create options, camera, scene, context
        self.vopt = mujoco.MjvOption()
        self.cam  = mujoco.MjvCamera()
        self.scn  = mujoco.MjvScene(self.model, maxgeom=self.maxgeom)
        self.pert = mujoco.MjvPerturb()

        # Context is initialized outside MuJoCoMinimalViewer class
        # self.ctx  = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        # set up rgb overlay
        width, height = glfw.get_framebuffer_size(self.window)

        # # Set window position
        # if x_offset is not None or y_offset is not None:
        #     # window size
        #     win_w,win_h = glfw.get_window_size(self.window)

        #     # default = center
        #     x_px = int((monitor_w-win_w)*0.5)
        #     y_px = int((monitor_h-win_h)*0.5)

        #     # x offset (ratio: monitor-based, else pixel)
        #     if x_offset is not None:
        #         if 0.0 <= x_offset <= 1.0:
        #             x_px = int(monitor_w*float(x_offset))
        #         else:
        #             x_px = int(x_offset)

        #     # y offset (ratio: monitor-based, else pixel)
        #     if y_offset is not None:
        #         if 0.0 <= y_offset <= 1.0:
        #             y_px = int(monitor_h*float(y_offset))
        #         else:
        #             y_px = int(y_offset)

        #     # clamp so window stays fully on screen
        #     x_px = max(0,min(x_px,monitor_w-win_w))
        #     y_px = max(0,min(y_px,monitor_h-win_h))

        #     glfw.set_window_pos(self.window,x_px,y_px)

        # Set window position
        win_w,win_h = glfw.get_window_size(self.window)

        # default = center
        x_px = int((monitor_w-win_w)*0.5)
        y_px = int((monitor_h-win_h)*0.5)

        # Offsets accept either monitor-relative ratios (0~1) or raw pixels.
        if x_offset is not None:
            if 0.0 <= x_offset <= 1.0:
                x_px = int(monitor_w*float(x_offset))
            else:
                x_px = int(x_offset)

        # Same convention for vertical placement.
        if y_offset is not None:
            if 0.0 <= y_offset <= 1.0:
                y_px = int(monitor_h*float(y_offset))
            else:
                y_px = int(y_offset)

        # clamp so window stays fully on screen
        x_px = max(0,min(x_px,monitor_w-win_w))
        y_px = max(0,min(y_px,monitor_h-win_h))

        glfw.set_window_pos(self.window,x_px,y_px)

        # get viewport
        self.viewport = mujoco.MjrRect(0, 0, framebuffer_width, framebuffer_height)

        # overlay, markers
        self._overlay = {}
        self._markers = []

        # rgb images to overlay
        self.rgb_overlay_top_right    = None
        self.rgb_overlay_top_left     = None
        self.rgb_overlay_bottom_right = None
        self.rgb_overlay_bottom_left  = None

        # Perturbation
        self.perturbation = perturbation

    def add_marker(self, **marker_params):
        """
        Add a marker to be rendered in the viewer.
        """
        self._markers.append(marker_params)

    def _add_marker_to_scene(self, marker):
        """
        Add a marker's properties to the MuJoCo scene.

        Parameters:
            marker (dict): A dictionary containing marker properties to be applied.
        """        
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError(
                'Ran out of geoms. maxgeom: %d' %
                self.scn.maxgeom)

        g = self.scn.geoms[self.scn.ngeom]

        # default values.
        g.dataid      = -1
        g.objtype     = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.objid       = -1
        g.category    = mujoco.mjtCatBit.mjCAT_DECOR
        g.matid       = -1 # newly added (by Jihwan, 2025-02-27)
        g.emission    = 0
        g.specular    = 0.5
        g.shininess   = 0.5
        g.reflectance = 0
        g.type        = mujoco.mjtGeom.mjGEOM_BOX
        g.size[:]     = np.ones(3) * 0.1
        g.mat[:]      = np.eye(3)
        g.rgba[:]     = np.ones(4)

        for key, value in marker.items():
            # setattr(g, key, value)
            if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
                setattr(g, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(g, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, str):
                # assert key == "label", "Only label is a string in mjtGeom."
                if value is None:
                    g.label[0] = 0
                else:
                    g.label = value
            elif hasattr(g, key):
                raise ValueError(
                    "mjtGeom has attr {} but type {} is invalid".format(
                        key, type(value)))
            else:
                raise ValueError("mjtGeom doesn't have field %s" % key)
            
        # Increment number of geoms
        self.scn.ngeom += 1
        return
    
    def apply_perturbations(self):
        """
        Apply any active perturbations to the simulation data by resetting external forces
        and applying the perturbation pose and force.

        Returns:
            None
        """        
        self.data.xfrc_applied = np.zeros_like(self.data.xfrc_applied)
        mujoco.mjv_applyPerturbPose(self.model, self.data, self.pert, 0)
        mujoco.mjv_applyPerturbForce(self.model, self.data, self.pert)

    def add_overlay(
            self,
            loc     = 'bottom left',
            gridpos = mujoco.mjtGridPos.mjGRID_TOPLEFT,
            text1   = '',
            text2   = '',
        ):
        """
        Add a text overlay to the viewer at a specified location.

        Parameters:
            loc (str): Location of the overlay ('top', 'top right', 'top left', 'bottom', 'bottom right', 'bottom left').
            gridpos: MuJoCo grid position constant for overlay placement.
            text1 (str): Primary text string.
            text2 (str): Secondary text string.

        Returns:
            None
        """
        if loc is not None:
            if loc == 'top': gridpos = mujoco.mjtGridPos.mjGRID_TOP
            elif loc == 'top right': gridpos = mujoco.mjtGridPos.mjGRID_TOPRIGHT
            elif loc == 'top left': gridpos = mujoco.mjtGridPos.mjGRID_TOPLEFT
            elif loc == 'bottom': gridpos = mujoco.mjtGridPos.mjGRID_BOTTOM
            elif loc == 'bottom right': gridpos = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT
            elif loc == 'bottom left': gridpos = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT
            
        if gridpos not in self._overlay:
            self._overlay[gridpos] = ["", ""]
            self._overlay[gridpos][0] += text1
            self._overlay[gridpos][1] += text2    
        else:
            self._overlay[gridpos][0] += "\n" + text1
            self._overlay[gridpos][1] += "\n" + text2                 

    def plot_rgb_overlay(self,rgb=None,loc='top right'):
        """
        Plot an RGB overlay at a specified location in the viewer window.

        Parameters:
            rgb (np.ndarray): The RGB image to overlay.
            loc (str): Location of the overlay. Options include 'top right', 'top left',
                'bottom right', or 'bottom left'.

        Returns:
            None
        """
        w_window,h_window = glfw.get_framebuffer_size(self.window)
        h_overlay,w_overlay = h_window//4,w_window//4
        # Fix aspect ratio
        h_raw,w_raw = rgb.shape[:2]
        # Calculate scale to preserve aspect ratio
        scale = min(w_overlay/w_raw,h_overlay/h_raw)
        w_new = int(w_raw*scale)
        h_new = int(h_raw*scale)
        # Resize
        rgb_resized = cv2.resize(rgb,(w_new,h_new),interpolation=cv2.INTER_NEAREST)
        # Create a black canvas with the target size
        rgb_padded = np.zeros((h_overlay,w_overlay,3),dtype=np.uint8)
        # Calculate the top-left corner for centering the resized image
        x_offset = (w_overlay-w_new) // 2
        y_offset = (h_overlay-h_new) // 2
        # Place the resized image onto the canvas
        rgb_padded[y_offset:y_offset+h_new,x_offset:x_offset+w_new] = rgb_resized
        # Store the RGB overlay
        if loc=='top right':
            self.rgb_overlay_top_right = rgb_padded
        elif loc=='top left':
            self.rgb_overlay_top_left = rgb_padded
        elif loc=='bottom right':
            self.rgb_overlay_bottom_right = rgb_padded
        elif loc=='bottom left':
            self.rgb_overlay_bottom_left = rgb_padded
        else:
            print ("Invalid location for RGB overlay. Use 'top right', 'top left', 'bottom right', or 'bottom left'.")

    def reset_rgb_overlay(self,loc=None):
        """
        Reset the RGB overlay(s).

        Parameters:
            loc (str, optional): Specific overlay location to reset. If None, all RGB overlays are reset.
                Valid locations are 'top right', 'top left', 'bottom right', or 'bottom left'.

        Returns:
            None
        """
        if loc is None:
            self.rgb_overlay_top_right    = None
            self.rgb_overlay_top_left     = None
            self.rgb_overlay_bottom_right = None
            self.rgb_overlay_bottom_left  = None
        else:
            if loc=='top right':
                self.rgb_overlay_top_right = None
            if loc=='top left':
                self.rgb_overlay_top_left = None
            if loc=='bottom right':
                self.rgb_overlay_bottom_right = None
            if loc=='bottom left':
                self.rgb_overlay_bottom_left = None

    def update_render_scene(self):
        """
        Render one viewer frame and process overlays.
        """
        render_start = time.time()
        width,height = glfw.get_framebuffer_size(self.window)
        self.viewport.width,self.viewport.height = width,height

        with self._gui_lock:
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                self.vopt,
                self.pert,
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,
                self.scn,
            )

            for marker in self._markers:
                self._add_marker_to_scene(marker)

            mujoco.mjr_render(self.viewport,self.scn,self.ctx)

            for gridpos,[t1,t2] in self._overlay.items():
                mujoco.mjr_overlay(
                    mujoco.mjtFontScale.mjFONTSCALE_100,
                    gridpos,
                    self.viewport,
                    t1,
                    t2,
                    self.ctx,
                )

            if self.rgb_overlay_top_right is not None:
                h_overlay,w_overlay = self.rgb_overlay_top_right.shape[:2]
                viewport_rgb_top_right = mujoco.MjrRect(
                    left   = 3*w_overlay,
                    bottom = 3*h_overlay,
                    width  = w_overlay,
                    height = h_overlay,
                )
                mujoco.mjr_drawPixels(
                    rgb      = np.flipud(self.rgb_overlay_top_right).flatten(),
                    depth    = None,
                    viewport = viewport_rgb_top_right,
                    con      = self.ctx,
                )
            if self.rgb_overlay_top_left is not None:
                h_overlay,w_overlay = self.rgb_overlay_top_left.shape[:2]
                viewport_rgb_top_left = mujoco.MjrRect(
                    left   = 0*w_overlay,
                    bottom = 3*h_overlay,
                    width  = w_overlay,
                    height = h_overlay,
                )
                mujoco.mjr_drawPixels(
                    rgb      = np.flipud(self.rgb_overlay_top_left).flatten(),
                    depth    = None,
                    viewport = viewport_rgb_top_left,
                    con      = self.ctx,
                )
            if self.rgb_overlay_bottom_right is not None:
                h_overlay,w_overlay = self.rgb_overlay_bottom_right.shape[:2]
                viewport_rgb_bottom_right = mujoco.MjrRect(
                    left   = 3*w_overlay,
                    bottom = 0*h_overlay,
                    width  = w_overlay,
                    height = h_overlay,
                )
                mujoco.mjr_drawPixels(
                    rgb      = np.flipud(self.rgb_overlay_bottom_right).flatten(),
                    depth    = None,
                    viewport = viewport_rgb_bottom_right,
                    con      = self.ctx,
                )
            if self.rgb_overlay_bottom_left is not None:
                h_overlay,w_overlay = self.rgb_overlay_bottom_left.shape[:2]
                viewport_rgb_bottom_left = mujoco.MjrRect(
                    left   = 0*w_overlay,
                    bottom = 0*h_overlay,
                    width  = w_overlay,
                    height = h_overlay,
                )
                mujoco.mjr_drawPixels(
                    rgb      = np.flipud(self.rgb_overlay_bottom_left).flatten(),
                    depth    = None,
                    viewport = viewport_rgb_bottom_left,
                    con      = self.ctx,
                )

            glfw.swap_buffers(self.window)

        glfw.poll_events()
        self._time_per_render = (
            0.9*self._time_per_render + 0.1*(time.time()-render_start)
        )

    def render(self,reset_rgb_overlay=True):
        """
        Render the current scene in the viewer window.

        Parameters:
            reset_rgb_overlay (bool): If True, resets the RGB overlay after rendering. This allows for dynamic updates to the overlay content on each render call.

        This method updates the scene, processes overlay items and figures, draws RGB overlays,
        swaps the buffers, and applies any active perturbations. It supports both paused and continuous
        rendering loops.
        """        
        if not self.is_alive:
            raise Exception(
                "GLFW window does not exist but you tried to render.")
        if glfw.window_should_close(self.window):
            self.close()
            return

        if self._paused: # if paused
            while self._paused:
                self.update_render_scene()
                if glfw.window_should_close(self.window):
                    self.close()
                    break
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
                # Resume using 'glfw.KEY_SPACE'
        else:
            self._loop_count += self.model.opt.timestep / \
                (self._time_per_render * self._run_speed)
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                self.update_render_scene()
                self._loop_count -= 1

        # clear markers
        self._markers[:] = []
        
        # clear overlay
        self._overlay.clear()
        if reset_rgb_overlay: 
            # clear rgb overlay
            self.reset_rgb_overlay()

        # apply perturbation (should this come before mj_step?)
        if self.perturbation:
            self.apply_perturbations()                

    def close(self):
        """
        Close the viewer, terminate the GLFW window, and free the MuJoCo context resources.
        """
        # if not self.is_alive:
        #     return
        # self.is_alive=False
        # try:
        #     if hasattr(self,"ctx") and self.ctx is not None:
        #         self.ctx.free()
        # except Exception:
        #     pass
        # try:
        #     if self.window is not None:
        #         glfw.destroy_window(self.window)
        # except Exception:
        #     pass
        # try:
        #     glfw.terminate()
        # except Exception:
        #     pass
        if not self.is_alive:
            return

        self.is_alive = False

        try:
            if self.window is not None:
                glfw.set_window_should_close(self.window,True)
                glfw.hide_window(self.window)
                glfw.make_context_current(self.window)
        except Exception:
            pass

        try:
            if hasattr(self,"ctx") and self.ctx is not None:
                self.ctx.free()
                # self.ctx = None # Don't set to None to avoid potential double free if close() is called multiple times
        except Exception:
            pass

        try:
            glfw.make_context_current(None)
        except Exception:
            pass

        try:
            if self.window is not None:
                glfw.destroy_window(self.window)
                self.window = None
        except Exception:
            pass

        try:
            glfw.poll_events()
        except Exception:
            pass


    # Viewer settings
    def set_transparency(self,transparent=True):
        """
        Set the transparency (alpha value) for rendering.
        """        
        self.vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = transparent

    def get_cam_info(self):
        """
        Get the current camera information.

        Returns:
            tuple: (azimuth,distance,elevation,lookat)
        """
        azimuth   = float(self.cam.azimuth)
        distance  = float(self.cam.distance)
        elevation = float(self.cam.elevation)
        lookat    = self.cam.lookat.copy()
        return azimuth,distance,elevation,lookat
    
    def set_cam_info(self,azimuth=None,distance=None,elevation=None,lookat=None):
        """ 
        Set the camera information.

        Parameters:
            azimuth (float): Azimuth angle.
            distance (float): Distance from the target.
            elevation (float): Elevation angle.
            lookat (list or np.ndarray): Look-at point coordinates.
        """
        self.cam.azimuth   = azimuth if azimuth is not None else self.cam.azimuth
        self.cam.distance  = distance if distance is not None else self.cam.distance
        self.cam.elevation = elevation if elevation is not None else self.cam.elevation
        if lookat is not None:
            self.cam.lookat[:] = np.array(lookat).reshape(self.cam.lookat.shape)

    def set_geomgroup(
            self,
            group_0 = True,
            group_1 = True,
            group_2 = True,
            group_3 = False,
            group_4 = None,
            group_5 = None,
        ):
        """
        Set the geometry group visibility flags.

        Geometry group convention for project MJCF/XML assets:
            group 0: floor geoms.
            group 1: normal visible colliders, such as boxes or cylinders whose
                visual and collision geometry are the same.
            group 2: visual-only geoms.
            group 3: collision-only geoms.

        When adding or updating XML assets, prefer explicit geom group values
        instead of relying on MuJoCo defaults. Also prefer mesh file paths that
        are relative to the XML file itself, rather than using a global meshdir,
        because merged XML files are easier to relocate and debug.

        A simple viewer pattern for checking groups is to bind keys 0-4 to
        geom groups and keys 5-9 to site groups via toggle_viewer_group(), then
        display the current state with get_viewer_group().

        Parameters:
            group_0 (bool): Visibility flag for geometry group 0.
            group_1 (bool): Visibility flag for geometry group 1.
            group_2 (bool): Visibility flag for geometry group 2.
            group_3 (bool): Visibility flag for geometry group 3.
            group_4 (bool): Visibility flag for geometry group 4.
            group_5 (bool): Visibility flag for geometry group 5.
        """
        if group_0 is not None: self.vopt.geomgroup[0] = group_0
        if group_1 is not None: self.vopt.geomgroup[1] = group_1
        if group_2 is not None: self.vopt.geomgroup[2] = group_2
        if group_3 is not None: self.vopt.geomgroup[3] = group_3
        if group_4 is not None: self.vopt.geomgroup[4] = group_4
        if group_5 is not None: self.vopt.geomgroup[5] = group_5

    def set_sitegroup(
            self,
            group_0 = None,
            group_1 = None,
            group_2 = None,
            group_3 = None,
            group_4 = None,
            group_5 = None,
        ):
        """
        Set the site group visibility flags.

        Parameters:
            group_0 (bool): Visibility flag for site group 0.
            group_1 (bool): Visibility flag for site group 1.
            group_2 (bool): Visibility flag for site group 2.
            group_3 (bool): Visibility flag for site group 3.
            group_4 (bool): Visibility flag for site group 4.
            group_5 (bool): Visibility flag for site group 5.
        """
        if group_0 is not None: self.vopt.sitegroup[0] = group_0
        if group_1 is not None: self.vopt.sitegroup[1] = group_1
        if group_2 is not None: self.vopt.sitegroup[2] = group_2
        if group_3 is not None: self.vopt.sitegroup[3] = group_3
        if group_4 is not None: self.vopt.sitegroup[4] = group_4
        if group_5 is not None: self.vopt.sitegroup[5] = group_5

    def set_convexhull(self,convexhull=False):
        """
        Set the visibility of convex hulls in the viewer.

        Parameters:
            convexhull (bool): If True, convex hulls are shown; otherwise, they are hidden.
        """
        self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = convexhull
        
# MuJoCoParser class
class MuJoCoParser(object):
    """
    MuJoCo model parser class to parse MuJoCo XML files and extract model information.
    """
    def __init__(
            self,
            name         = None,
            rel_xml_path = None,
            verbose      = False,
        ):
        """
        Initialize
        
        Parameters:
            name (str): Name of the MuJoCo model
            rel_xml_path (str): Relative path to the MuJoCo XML file
        """
        # Store parameters
        self.name         = name
        self.rel_xml_path = rel_xml_path
        self.verbose      = verbose

        # Parse XML
        self._parse_xml(self.rel_xml_path)

        # Set name if not provided
        if self.name is None: self.name = self.model_name

        # Get monitor size
        self.monitor_width,self.monitor_height = get_monitor_size()

        # Flags
        self.use_mujoco_viewer = False

        # Timing variables used in self.is_viewer_alive() method
        self._last_alive_t: Optional[float] = None
        self.loop_dt: Optional[float]       = 0.0
        self.loop_hz: Optional[float]       = 0.0
        self.loop_hz_ema: Optional[float]   = 0.0
        self._loop_hz_ema_alpha: float      = 0.05

        # Initial joint positions
        self.q0       = self.get_qpos()
        self.qrev0    = self.get_qpos(joint_names=self.rev_joint_names,flatten=True)
        self.qactive0 = self.get_qpos(joint_names=self.active_joint_names,flatten=True)

        # Final RGB image
        self.final_rgb_img = None

        # Object point cloud cache (body-local canonical pcd)
        self.obj_pcd_local_cache = {}

        # TicToc instance
        self.tt = TicToc()

        # Reset
        self.reset()

        # Print information
        if self.verbose:
            self.print_info()
        else:
            print ("MuJoCo model loaded from [%s]."%(self.rel_xml_path))

    def _parse_xml(self,rel_xml_path):
        """
        Parse the MuJoCo XML file

        Parameters:
            rel_xml_path (str): Relative path to the MuJoCo XML file
        """
        self.full_xml_path = os.path.abspath(os.path.join(os.getcwd(),rel_xml_path))
        self.model         = mujoco.MjModel.from_xml_path(self.full_xml_path)

        # Parse xml model name
        parsed_strings = [s for s in self.model.names.split(b'\x00') if s] 
        parsed_strings = [s.decode('utf-8') for s in parsed_strings]
        self.model_name = parsed_strings[0]

        # Create MjData instance
        self.data = mujoco.MjData(self.model)
        self.dt   = self.model.opt.timestep
        self.HZ   = int(1/self.dt)

        # Parse integrator type
        """
        MuJoCo integrator details can be found here:
        https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjtintegrator
        """
        self.integrator = self.model.opt.integrator
        if self.integrator == mujoco.mjtIntegrator.mjINT_EULER:
            self.integrator_name = 'EULER'
        elif self.integrator == mujoco.mjtIntegrator.mjINT_RK4:
            self.integrator_name = 'RK4'
        elif self.integrator == mujoco.mjtIntegrator.mjINT_IMPLICIT:
            self.integrator_name = 'IMPLICIT'
        elif self.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
            self.integrator_name = 'IMPLICITFAST'
        else:
            self.integrator_name = 'UNKNOWN'

        # Parse model dimensions (I prefer to use snake_case for variable names)
        self.n_q = self.model.nq  # number of generalized coordinates
        self.n_v = self.model.nv  # number of generalized velocities
        self.n_u = self.model.nu  # number of actuators

        # Geometry information
        self.n_geom           = self.model.ngeom # number of geometries
        self.geom_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,geom_idx) 
                                 for geom_idx in range(self.n_geom)]
        
        # Mesh information
        self.n_mesh           = self.model.nmesh # number of meshes
        self.mesh_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_MESH,mesh_idx)
                              for mesh_idx in range(self.n_mesh)]

        # Body information
        self.n_body           = self.model.nbody # number of bodies
        self.body_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,body_idx)
                                 for body_idx in range(self.n_body)]
        self.body_masses      = self.model.body_mass
        self.body_total_mass  = self.body_masses.sum()
        # Keep name-level parent lookup because many helpers operate on body
        # names rather than integer body ids.
        self.parent_body_names = [] # parent body names
        for b_idx in range(self.n_body):
            parent_id = self.model.body_parentid[b_idx]
            parent_body_name = self.body_names[parent_id]
            self.parent_body_names.append(parent_body_name)

        # Joint information
        self.n_joint          = self.model.njnt # number of joints
        self.joint_names      = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_JOINT,joint_idx)
                                 for joint_idx in range(self.n_joint)]
        self.joint_types      = self.model.jnt_type # joint types
        self.joint_ranges     = self.model.jnt_range # joint ranges
        self.joint_mins       = self.joint_ranges[:,0]
        self.joint_maxs       = self.joint_ranges[:,1]

        # Free joint
        self.free_joint_idxs  = np.where(self.joint_types==mujoco.mjtJoint.mjJNT_FREE)[0].astype(np.int32)
        self.free_joint_names = [self.joint_names[joint_idx] for joint_idx in self.free_joint_idxs]
        self.n_free_joint     = len(self.free_joint_idxs)

        # Revolute Joint
        self.rev_joint_idxs   = np.where(self.joint_types==mujoco.mjtJoint.mjJNT_HINGE)[0].astype(np.int32)
        self.rev_joint_names  = [self.joint_names[joint_idx] for joint_idx in self.rev_joint_idxs]
        self.n_rev_joint      = len(self.rev_joint_idxs)
        self.rev_joint_mins   = self.joint_ranges[self.rev_joint_idxs,0]
        self.rev_joint_maxs   = self.joint_ranges[self.rev_joint_idxs,1]
        self.rev_joint_ranges = self.rev_joint_maxs - self.rev_joint_mins

        # Prismatic Joint
        self.pri_joint_idxs   = np.where(self.joint_types==mujoco.mjtJoint.mjJNT_SLIDE)[0].astype(np.int32)
        self.pri_joint_names  = [self.joint_names[joint_idx] for joint_idx in self.pri_joint_idxs]
        self.n_pri_joint      = len(self.pri_joint_idxs)
        self.pri_joint_mins   = self.joint_ranges[self.pri_joint_idxs,0]
        self.pri_joint_maxs   = self.joint_ranges[self.pri_joint_idxs,1]
        self.pri_joint_ranges = self.pri_joint_maxs - self.pri_joint_mins

        # Revolute + prismatic Joint Information
        self.n_rev_pri_joint      = self.n_rev_joint + self.n_pri_joint
        self.rev_pri_joint_idxs   = np.concatenate([self.rev_joint_idxs,self.pri_joint_idxs])
        self.rev_pri_joint_names  = self.rev_joint_names + self.pri_joint_names
        self.rev_pri_joint_mins   = np.concatenate([self.rev_joint_mins,self.pri_joint_mins])
        self.rev_pri_joint_maxs   = np.concatenate([self.rev_joint_maxs,self.pri_joint_maxs])
        self.rev_pri_joint_ranges = self.rev_pri_joint_maxs - self.rev_pri_joint_mins

        # Equality-constrained revolute joints are treated as passive so
        # downstream helpers can target only independently driven joints.
        self.n_eq = self.model.eq_data.shape[0] # number of equality constraints
        self.active_joint_names = self.rev_pri_joint_names.copy()
        self.passive_joint_names = []
        for eq_idx in range(self.n_eq): # for each joint equality constraints
            if self.model.eq_type[eq_idx] == mujoco.mjtEq.mjEQ_JOINT: # joint equality constraint
                joint1 = self.model.joint(self.model.eq_obj1id[eq_idx]).name # passive joint name
                # If the passive joint is in the active joint list, move it to the passive joint list
                if joint1 in self.active_joint_names:
                    self.active_joint_names.remove(joint1)
                    self.passive_joint_names.append(joint1)
        self.active_rev_joint_names = [j for j in self.active_joint_names if j in self.rev_joint_names]
        self.active_rev_pri_joint_names = self.active_joint_names.copy()
        self.n_active_joint  = len(self.active_joint_names)
        self.active_joint_mins = self.joint_mins[get_idxs(self.joint_names,self.active_joint_names)]
        self.active_joint_maxs = self.joint_maxs[get_idxs(self.joint_names,self.active_joint_names)]
        self.active_joint_ranges = self.active_joint_maxs - self.active_joint_mins
        self.n_passive_joint = len(self.passive_joint_names)
        self.passive_joint_mins = self.joint_mins[get_idxs(self.joint_names,self.passive_joint_names)]
        self.passive_joint_maxs = self.joint_maxs[get_idxs(self.joint_names,self.passive_joint_names)]
        self.passive_joint_ranges = self.passive_joint_maxs - self.passive_joint_mins

        # Actuator (or control) information
        self.n_ctrl           = self.model.nu # number of actuators (or controls)
        self.ctrl_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,ctrl_idx) 
                                 for ctrl_idx in range(self.n_ctrl)]
        self.ctrl_ranges      = self.model.actuator_ctrlrange # control range
        self.ctrl_mins        = self.ctrl_ranges[:,0]
        self.ctrl_maxs        = self.ctrl_ranges[:,1]
        self.ctrl_gears       = self.model.actuator_gear[:,0] # gears
        self.ctrl_trntypes    = self.model.actuator_trntype.copy()
        self.ctrl_gaintypes   = self.model.actuator_gaintype.copy()
        self.ctrl_biastypes   = self.model.actuator_biastype.copy()
        self.actuator_gainprm_0    = self.model.actuator_gainprm.copy()
        self.actuator_biasprm_0    = self.model.actuator_biasprm.copy()
        self.actuator_ctrlrange_0  = self.model.actuator_ctrlrange.copy()
        self.actuator_forcerange_0 = self.model.actuator_forcerange.copy()

        # Camera information
        self.n_cam            = self.model.ncam
        self.cam_names        = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_CAMERA,cam_idx) 
                                 for cam_idx in range(self.n_cam)]
        self.cams             = []
        self.cam_fovs         = []
        self.cam_viewports    = []
        for cam_idx in range(self.n_cam):
            cam_name = self.cam_names[cam_idx]
            cam      = mujoco.MjvCamera()
            cam.fixedcamid = self.model.cam(cam_name).id
            cam.type       = mujoco.mjtCamera.mjCAMERA_FIXED
            cam_fov        = self.model.cam_fovy[cam_idx]
            viewport       = mujoco.MjrRect(0,0,800,600) # SVGA?
            # Append
            self.cams.append(cam)
            self.cam_fovs.append(cam_fov)
            self.cam_viewports.append(viewport)

        # Build actuator-centric joint metadata once so later helpers do not
        # repeatedly traverse MuJoCo transmission fields.
        self.ctrl_qpos_idxs  = [] # qpos index attached to control, -1 if not attached to any joint
        self.ctrl_qpos_names = [] # qpos name attached to control, empty string if not attached to any joint
        self.ctrl_qpos_mins  = [] # minimum qpos value attached to control
        self.ctrl_qpos_maxs  = [] # maximum qpos value attached to control
        self.ctrl_qvel_idxs  = [] # qvel index attached to control, -1 if not attached to any joint
        self.ctrl_types      = [] # type of control (JOINT, TENDON, UNKNOWN)

        for ctrl_idx in range(self.n_ctrl):
            trntype = self.model.actuator_trntype[ctrl_idx]

            if trntype == mujoco.mjtTrn.mjTRN_JOINT:
                joint_idx = self.model.actuator(self.ctrl_names[ctrl_idx]).trnid[0]

                self.ctrl_qpos_idxs.append(self.model.jnt_qposadr[joint_idx])
                self.ctrl_qpos_names.append(self.joint_names[joint_idx])
                self.ctrl_qpos_mins.append(self.joint_ranges[joint_idx,0])
                self.ctrl_qpos_maxs.append(self.joint_ranges[joint_idx,1])
                self.ctrl_qvel_idxs.append(self.model.jnt_dofadr[joint_idx])
                self.ctrl_types.append('JOINT')

            elif trntype == mujoco.mjtTrn.mjTRN_TENDON:
                self.ctrl_qpos_idxs.append(-1)
                self.ctrl_qpos_names.append('')
                self.ctrl_qpos_mins.append(np.nan)
                self.ctrl_qpos_maxs.append(np.nan)
                self.ctrl_qvel_idxs.append(-1)
                self.ctrl_types.append('TENDON')

            else:
                self.ctrl_qpos_idxs.append(-1)
                self.ctrl_qpos_names.append('')
                self.ctrl_qpos_mins.append(np.nan)
                self.ctrl_qpos_maxs.append(np.nan)
                self.ctrl_qvel_idxs.append(-1)
                self.ctrl_types.append('UNKNOWN (trntype=%d)'%(trntype))

        # Semantic aliases for control to joint mapping
        self.ctrl_joint_names = self.ctrl_qpos_names.copy()
        self.ctrl_joint_mins  = self.ctrl_qpos_mins.copy()
        self.ctrl_joint_maxs  = self.ctrl_qpos_maxs.copy()

        # Mirror the mapping in the opposite direction because notebook code
        # often starts from a joint selection and asks which actuators drive it.
        self.joint_ctrl_idxs   = [] # control indices attached to each joint
        self.joint_ctrl_names  = [] # control names attached to each joint
        self.joint_ctrl_qpos_idxs = [] # qpos indices attached to each joint through controls
        self.joint_ctrl_qvel_idxs = [] # qvel indices attached to each joint through controls
        self.joint_ctrl_types  = [] # control types attached to each joint

        for joint_idx in range(self.n_joint):
            ctrl_idxs_for_joint      = []
            ctrl_names_for_joint     = []
            ctrl_qpos_idxs_for_joint = []
            ctrl_qvel_idxs_for_joint = []
            ctrl_types_for_joint     = []

            for ctrl_idx in range(self.n_ctrl):
                trntype = self.model.actuator_trntype[ctrl_idx]

                # Only joint transmission is directly mapped to a joint
                if trntype != mujoco.mjtTrn.mjTRN_JOINT:
                    continue

                attached_joint_idx = self.model.actuator(self.ctrl_names[ctrl_idx]).trnid[0]

                if attached_joint_idx == joint_idx:
                    ctrl_idxs_for_joint.append(ctrl_idx)
                    ctrl_names_for_joint.append(self.ctrl_names[ctrl_idx])
                    ctrl_qpos_idxs_for_joint.append(self.model.jnt_qposadr[joint_idx])
                    ctrl_qvel_idxs_for_joint.append(self.model.jnt_dofadr[joint_idx])
                    ctrl_types_for_joint.append('JOINT')

            self.joint_ctrl_idxs.append(ctrl_idxs_for_joint)
            self.joint_ctrl_names.append(ctrl_names_for_joint)
            self.joint_ctrl_qpos_idxs.append(ctrl_qpos_idxs_for_joint)
            self.joint_ctrl_qvel_idxs.append(ctrl_qvel_idxs_for_joint)
            self.joint_ctrl_types.append(ctrl_types_for_joint)

        # Position-style actuator subset
        # Practical rule:
        # - joint transmission only
        # - non-zero gain term
        # - non-zero q-position bias term
        # This excludes common velocity actuators that usually use only qvel bias.
        # Tuning helpers use this subset to avoid touching tendon actuators or
        # non-position-style controllers by default.
        self.position_ctrl_idxs        = []
        self.position_ctrl_names       = []
        self.position_ctrl_joint_names = []
        for ctrl_idx in range(self.n_ctrl):
            if self.ctrl_types[ctrl_idx] != 'JOINT':
                continue

            gainprm = np.asarray(self.model.actuator_gainprm[ctrl_idx],dtype=np.float64)
            biasprm = np.asarray(self.model.actuator_biasprm[ctrl_idx],dtype=np.float64)

            is_position_style = (
                np.isfinite(gainprm[0]) and
                (np.abs(gainprm[0]) > 0.0) and
                (np.abs(biasprm[1]) > 0.0)
            )
            if not is_position_style:
                continue

            self.position_ctrl_idxs.append(ctrl_idx)
            self.position_ctrl_names.append(self.ctrl_names[ctrl_idx])
            self.position_ctrl_joint_names.append(self.ctrl_joint_names[ctrl_idx])

        # Sensor information
        self.n_sensor      = self.model.nsensor
        self.sensor_names  = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_SENSOR,sensor_idx)
                              for sensor_idx in range(self.n_sensor)]
        
        # Site information
        self.n_site        = self.model.nsite
        self.site_names    = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_SITE,site_idx)
                              for site_idx in range(self.n_site)]

        # Geometry original parameters
        self.geom_friction_0 = self.model.geom_friction.copy()
        
        # Cache the body tree once so subtree queries stay cheap and name-based.
        self.body_children_list = [[] for _ in range(self.model.nbody)]
        for cid in range(1,self.model.nbody):
            pid = int(self.model.body_parentid[cid])
            self.body_children_list[pid].append(cid)
        
    def print_info(self):
        """
        Print out a comprehensive summary of the model's configuration.
        """
        print("")
        print("-----------------------------------------------------------------------------")
        print("name:[%s] dt:[%.3f] HZ:[%d]" % (self.model_name, self.dt, self.HZ))
        print(" n_q:[%d] n_v:[%d] n_u:[%d]" % (self.n_q, self.n_v, self.n_u))
        print(" integrator:[%s]" % (self.integrator_name))

        # ------------------------------------------------------------------
        # Bodies
        # ------------------------------------------------------------------
        print("")
        print("Body information. n_body:[%d]" % (self.n_body))
        for body_idx, body_name in enumerate(self.body_names):
            body_mass = self.body_masses[body_idx]
            parent_name = self.parent_body_names[body_idx]
            print(" [%d/%d] [%s] parent:[%s] mass:[%.2f]kg" %
                (body_idx, self.n_body, body_name, parent_name, body_mass))
        print("body_total_mass:[%.2f]kg" % (self.body_total_mass))

        # ------------------------------------------------------------------
        # Geometry
        # ------------------------------------------------------------------
        print("")
        print("Geometry information. n_geom:[%d]" % (self.n_geom))
        for geom_idx, geom_name in enumerate(self.geom_names):
            body_id = self.model.geom_bodyid[geom_idx]
            body_name = self.body_names[body_id]
            print(" [%d/%d] [%s] attached_body:[%s]" %
                (geom_idx, self.n_geom, geom_name, body_name))

        # ------------------------------------------------------------------
        # Mesh
        # ------------------------------------------------------------------
        print("")
        print("Mesh information. n_mesh:[%d]" % (self.n_mesh))
        for mesh_idx, mesh_name in enumerate(self.mesh_names):
            print(" [%d/%d] [%s]" %
                (mesh_idx, self.n_mesh, mesh_name))

        # ------------------------------------------------------------------
        # Joints (all)
        # ------------------------------------------------------------------
        print("")
        print("Joint information. n_joint:[%d]" % (self.n_joint))
        for joint_idx, joint_name in enumerate(self.joint_names):
            axis = self.model.joint(joint_idx).axis
            print(" [%d/%d] [%s] axis:%s" %
                (joint_idx, self.n_joint, joint_name, axis))

        # ------------------------------------------------------------------
        # Free joints
        # ------------------------------------------------------------------
        print("\nFree joint information. n_free_joint:[%d]" % (self.n_free_joint))
        for idx, joint_name in enumerate(self.free_joint_names):
            body_id = self.model.joint(self.free_joint_idxs[idx]).bodyid[0]
            body_name = self.body_names[body_id]
            print(" [%d/%d] [%s] body_attached:[%s]" %
                (idx, self.n_free_joint, joint_name, body_name))

        # ------------------------------------------------------------------
        # Revolute joints
        # ------------------------------------------------------------------
        print("\nRevolute joint information. n_rev_joint:[%d]" % (self.n_rev_joint))
        for idx, joint_name in enumerate(self.rev_joint_names):
            print(" [%d/%d] [%s] range:[%.3f]~[%.3f]" %
                (idx, self.n_rev_joint, joint_name,
                self.rev_joint_mins[idx], self.rev_joint_maxs[idx]))

        # ------------------------------------------------------------------
        # Prismatic joints
        # ------------------------------------------------------------------
        print("\nPrismatic joint information. n_pri_joint:[%d]" % (self.n_pri_joint))
        for idx, joint_name in enumerate(self.pri_joint_names):
            print(" [%d/%d] [%s] range:[%.3f]~[%.3f]" %
                (idx, self.n_pri_joint, joint_name,
                self.pri_joint_mins[idx], self.pri_joint_maxs[idx]))

        # ------------------------------------------------------------------
        # Revolute + Prismatic joints
        # ------------------------------------------------------------------
        print("\nRevolute+Prismatic joint information. n_rev_pri_joint:[%d]" % (self.n_rev_pri_joint))
        for idx, joint_name in enumerate(self.rev_pri_joint_names):
            print(" [%d/%d] [%s] range:[%.3f]~[%.3f]" %
                (idx, self.n_rev_pri_joint, joint_name,
                self.rev_pri_joint_mins[idx], self.rev_pri_joint_maxs[idx]))

        # ------------------------------------------------------------------
        # Active / Passive joints
        # ------------------------------------------------------------------
        print("\nActive joint information. n_active_joint:[%d]" % (self.n_active_joint))
        for idx, joint_name in enumerate(self.active_joint_names):
            print(" [%d/%d] [%s] range:[%.3f]~[%.3f]" %
                (idx, self.n_active_joint, joint_name,
                self.active_joint_mins[idx], self.active_joint_maxs[idx]))

        print("\nPassive joint information. n_passive_joint:[%d]" % (self.n_passive_joint))
        for idx, joint_name in enumerate(self.passive_joint_names):
            print(" [%d/%d] [%s] range:[%.3f]~[%.3f]" %
                (idx, self.n_passive_joint, joint_name,
                self.passive_joint_mins[idx], self.passive_joint_maxs[idx]))

        # ------------------------------------------------------------------
        # Actuators / Controls
        # ------------------------------------------------------------------
        print("\nControl information. n_ctrl:[%d]" % (self.n_ctrl))
        for idx, ctrl_name in enumerate(self.ctrl_names):
            print(" [%d/%d] [%s] range:[%.3f]~[%.3f] gear:[%.2f] type:[%s]" %
                (idx, self.n_ctrl, ctrl_name,
                self.ctrl_mins[idx], self.ctrl_maxs[idx],
                self.ctrl_gears[idx], self.ctrl_types[idx]))

        # ------------------------------------------------------------------
        # Control-to-joint mapping
        # ------------------------------------------------------------------
        print("\nControl-to-joint mapping.")
        for idx in range(self.n_ctrl):
            print(" [%d/%d] ctrl:[%s] qpos_idx:[%d] qpos_name:[%s] qpos_range:[%.3f]~[%.3f] qvel_idx:[%d] type:[%s]" %
                (idx, self.n_ctrl,
                self.ctrl_names[idx],
                self.ctrl_qpos_idxs[idx],
                self.ctrl_qpos_names[idx],
                self.ctrl_qpos_mins[idx],
                self.ctrl_qpos_maxs[idx],
                self.ctrl_qvel_idxs[idx],
                self.ctrl_types[idx]))
            
        # ------------------------------------------------------------------
        # Joint-to-control mapping
        # ------------------------------------------------------------------
        print("\nJoint-to-control mapping.")
        for idx in range(self.n_joint):
            print(" [%d/%d] joint:[%s] ctrl_idxs:%s ctrl_names:%s" %
                (idx, self.n_joint,
                self.joint_names[idx],
                self.joint_ctrl_idxs[idx],
                self.joint_ctrl_names[idx]))            

        # ------------------------------------------------------------------
        # Cameras
        # ------------------------------------------------------------------
        print("\nCamera information. n_cam:[%d]" % (self.n_cam))
        for idx, cam_name in enumerate(self.cam_names):
            print(" [%d/%d] [%s] fov:[%.1f] viewport:%s" %
                (idx, self.n_cam, cam_name, self.cam_fovs[idx], self.cam_viewports[idx]))

        # ------------------------------------------------------------------
        # Sensors
        # ------------------------------------------------------------------
        print("")
        print("Sensor information. n_sensor:[%d]" % (self.n_sensor))
        for idx, sensor_name in enumerate(self.sensor_names):
            print(" [%d/%d] [%s]" % (idx, self.n_sensor, sensor_name))

        # ------------------------------------------------------------------
        # Sites
        # ------------------------------------------------------------------
        print("")
        print("Site information. n_site:[%d]" % (self.n_site))
        for idx, site_name in enumerate(self.site_names):
            print(" [%d/%d] [%s]" % (idx, self.n_site, site_name))

        print("-----------------------------------------------------------------------------")

    def reset(self,forward=True,step=False,clear_cache=True):
        """
        Reset the MuJoCo simulation data to the initial state.

        Parameters:
            forward (bool): If True, perform a forward pass after resetting.
            step (bool): If True, perform a simulation step after resetting.
        """
        time.sleep(1e-3) # add some sleep?
        mujoco.mj_resetData(self.model,self.data) # reset data
        if forward:
            # self.data.qpos = self.model.qpos0.copy()
            self.data.qpos[:] = self.q0
            mujoco.mj_forward(self.model,self.data)
        if step:
            mujoco.mj_step(self.model,self.data) # step once to update

        # Reset wall time
        self.init_sim_time    = self.data.time
        self.init_wall_time   = time.time()
        self.accum_wall_time  = 0.0  # accumulated wall time
        self.last_wall_update = time.time()

        # Reset double click info
        self.xyz_left_double_click  = None 
        self.xyz_right_double_click = None

        # Clear object pointcloud cache
        if clear_cache:
            self.clear_obj_pcd_cache()

        # Reset final RGB image
        self.final_rgb_img = None

    def init_viewer(
            self,
            title            = None,
            width            = 0.6,
            height           = 1.0,
            maxgeom          = 50000,
            perturbation     = True,
            fontscale        = 200,
            x_offset         = None,
            y_offset         = None,
        ):
        """
        Initialize the MuJoCo viewer.

        Parameters:
            title (str): Title of the viewer window.
            width (int or float): Width of the viewer window in pixels or relative to monitor width.
            height (int or float): Height of the viewer window in pixels or relative to monitor height.
            maxgeom (int): Maximum number of geometries to render.
            perturbation (bool): If True, enable perturbation mode.
            fontscale (int): Font scale for the viewer.
            x_offset (int): X offset of the viewer window.
            y_offset (int): Y offset of the viewer window.

        Defaults attributes:
            - Transparency: Enabled
            - Camera: Azimuth=173, Distance=3.56, Elevation=-22.56, Lookat=[0,0,0.59]
            - Geometry Groups: group_0=True, group_1=True, group_2=True, group_3=False 
            - Site Groups: group_0=False, group_1=False 

        Usual convention for geom groups:
            - group_0: floor or ground
            - group_1: visual and collision (e.g., box or cylinder)
            - group_2: visual only (e.g., accurate mesh) => contype="0" conaffinity="0"
            - group_3: collision only (e.g., simplified collision mesh) <= viewer off by default
        """

        # Create viewer instance
        self.use_mujoco_viewer = True

        # Set title
        if title is None: title = self.name

        # Notebook code usually passes monitor-relative ratios, while scripts may
        # pass absolute pixels. Support both without changing the API.
        if width <= 1.0 and height <= 1.0:
            w_monitor,h_monitor = get_monitor_size()
            width  = int(width*w_monitor)
            height = int(height*h_monitor)

        # Initialize viewer
        self.viewer = MuJoCoMinimalViewer(
            self.model,
            self.data,
            mode              = 'window',
            title             = title,
            width             = width,
            height            = height,
            maxgeom           = maxgeom,
            perturbation      = perturbation,
            x_offset          = x_offset,
            y_offset          = y_offset,
        )
        self.viewer.ctx = mujoco.MjrContext(self.model,fontscale)

        # Set some default viewer attributes
        self.viewer.set_transparency(transparent=False)
        self.viewer.set_cam_info(azimuth=173,distance=3.56,elevation=-22.56,lookat=[0,0,0.59])

        """
        Usual convention for geom groups:
            - group_0: floor or ground
            - group_1: visual and collision (e.g., box or cylinder)
            - group_2: visual only (e.g., accurate mesh) => contype="0" conaffinity="0"
            - group_3: collision only (e.g., simplified collision mesh) <= viewer off by default
        """
        self.viewer.set_geomgroup(group_0=True,group_1=True,group_2=True,group_3=False)

        # Hide site group 0 and 1 by default
        self.viewer.set_sitegroup(group_0=False,group_1=False)

    def is_viewer_alive(self):
        """
        Check if the viewer window is active.

        Returns:
            bool: True if viewer is alive, False otherwise.
        """
        if self.viewer is None: 
            return False

        # Honor GLFW close requests immediately even when render() is throttled.
        # Without this, pressing ESC can leave the outer loop spinning until the
        # next scheduled render call notices window_should_close().
        try:
            if (
                getattr(self.viewer,"is_alive",False)
                and getattr(self.viewer,"window",None) is not None
                and glfw.window_should_close(self.viewer.window)
            ):
                self.close_viewer()
                return False
        except Exception:
            pass
        
        # Update loop timing
        """ 
        Update loop timing information for performance monitoring.
            - loop_dt: Time difference between current and last alive check.
            - loop_hz: Raw frequency (Hz) calculated from loop_dt.
            - loop_hz_ema: Exponentially moving average of loop_hz for smoother tracking.
        """
        now = time.perf_counter()
        if self._last_alive_t is not None:
            self.loop_dt = now - self._last_alive_t
            if self.loop_dt > 0.0:
                # raw Hz
                self.loop_hz = 1.0 / self.loop_dt
                # EMA Hz
                if self.loop_hz_ema == 0.0:
                    self.loop_hz_ema = self.loop_hz
                else:
                    a = self._loop_hz_ema_alpha
                    self.loop_hz_ema = (1.0 - a) * self.loop_hz_ema + a * self.loop_hz
        self._last_alive_t = now        

        # Return alive status
        return self.viewer.is_alive
    
    def close_viewer(self):
        """
        Close the MuJoCo viewer and release resources.

        Returns:
            None
        """
        # self.use_mujoco_viewer = False
        # # Capture final RGB image
        # if self.final_rgb_img is None:
        #     self.final_rgb_img,_ = self.grab_current_rgbd_imgs()
        # # If viewer attribute does not exist
        # if not hasattr(self,'viewer'):
        #     return 
        # # If viewer is not initialized
        # if self.viewer is None: 
        #     return
        # # Close viewer
        # self.viewer.close()
        self.use_mujoco_viewer = False
        if not hasattr(self,"viewer"): # if viewer attribute does not exist,
            return
        if self.viewer is None: # if viewer is not initialized,
            return
        if not getattr(self.viewer,"is_alive",False): # if viewer is not alive,
            # self.viewer = None
            return
        if self.final_rgb_img is None: # if final RGB image is not captured yet,
            self.final_rgb_img,_ = self.capture_clean_viewer_rgbd_imgs()
        self.viewer.close()
        # self.viewer = None

    def capture_clean_viewer_rgbd_imgs(self):
        """
        Render one clean viewer frame without text/RGB overlays and capture it.

        This is mainly used for final-image capture so that persistent framebuffer
        contents from previous RGB overlays do not appear in the saved image.

        Returns:
            tuple: (rgb_img, depth_img)
        """
        if (
            not hasattr(self,"viewer")
            or self.viewer is None
            or not getattr(self.viewer,"is_alive",False)
        ):
            raise RuntimeError("[capture_clean_viewer_rgbd_imgs] Viewer is not alive.")

        window = self.viewer.window
        should_close_prev = False
        try:
            should_close_prev = glfw.window_should_close(window)
        except Exception:
            pass

        # Clear transient overlays first; otherwise old framebuffer content can
        # leak into the final screenshot.
        if hasattr(self.viewer,"_markers") and self.viewer._markers is not None:
            self.viewer._markers[:] = []
        if hasattr(self.viewer,"_overlay") and self.viewer._overlay is not None:
            self.viewer._overlay.clear()
        if hasattr(self.viewer,"reset_rgb_overlay"):
            self.viewer.reset_rgb_overlay()

        try:
            glfw.set_window_should_close(window,False)
        except Exception:
            pass

        self.viewer.render()
        rgb_img,depth_img = self.grab_current_rgbd_imgs()

        try:
            glfw.set_window_should_close(window,should_close_prev)
        except Exception:
            pass

        return rgb_img,depth_img

    def viewer_text_overlay(
            self,
            text1 = '',
            text2 = '',
            loc   = 'bottom left',
        ):
        """
        Add a text overlay on the viewer window.

        Parameters:
            loc (str): Overlay location ('top', 'top right', 'top left', 'bottom', 'bottom right', 'bottom left').
            text1 (str): Primary text.
            text2 (str): Secondary text.
        """
        self.viewer.add_overlay(loc=loc,text1=text1,text2=text2)

    def render(self, reset_rgb_overlay=True):
        """
        Render the current simulation state to the viewer.

        Parameters:
            reset_rgb_overlay (bool): If True, clear any existing RGB overlays before rendering.
        """
        # Preserve the last clean frame before shutdown so notebooks can inspect
        # the terminal state after the viewer loop exits.
        if glfw.KEY_ESCAPE in self.get_key_pressed_list():
            print ("Final RGB image captured.")
            if self.final_rgb_img is None:
                self.final_rgb_img,_ = self.capture_clean_viewer_rgbd_imgs()
        
        # Render
        if self.use_mujoco_viewer:
            # Assert
            assert (
                hasattr(self,"viewer")
                and self.viewer is not None
                and getattr(self.viewer,"is_alive",False)
            ), "[%s] Mujoco viewer is not alive."%(self.name)
            
            # Render
            self.viewer.render(reset_rgb_overlay=reset_rgb_overlay)
        else:
            # Viewer not initialized
            print ("[%s] Viewer NOT initialized."%(self.name))          

    def step(
            self,
            ctrl          = None,
            ctrl_idxs     = None,
            ctrl_names    = None,
            joint_names   = None,
            nstep         = 1,
            step_flag     = True,
        ):
        """ 
        Advance the simulation by a certain number of steps, optionally applying control inputs.

        Parameters:
            ctrl (np.array, optional): Control inputs to apply.
            ctrl_idxs (list, optional): Indices of controls to apply (overrides ctrl_names if not None).
            ctrl_names (list, optional): Names of controls to apply.
            joint_names (list, optional): Names of joints to apply control inputs to (used if ctrl_names is None).
            nstep (int): Number of simulation steps to advance.
            step_flag (bool): If True, perform the simulation step; if False, only update wall time without stepping.
        """
        # Apply control inputs and step simulation
        if step_flag:
            if ctrl is not None:
                # determine control indices
                if ctrl_names is not None: ctrl_idxs = get_idxs(self.ctrl_names,ctrl_names)
                elif joint_names is not None: ctrl_idxs = self.get_idxs_step(joint_names=joint_names)
                # set control inputs
                if ctrl_idxs is None:  self.data.ctrl[:] = ctrl
                else: self.data.ctrl[ctrl_idxs] = ctrl
            # Forward simulation
            mujoco.mj_step(self.model,self.data,nstep=nstep)

        # Update wall time
        self.increase_wall_time(step_flag=step_flag)

    def forward(
            self,
            q             = None,
            joint_names   = None,
            joint_idxs    = None,
            clip_position = True,
        ):
        """
        Perform forward kinematics, optionally updating the state qpos.

        Parameters:
            q (np.array, optional): New joint positions to set.
            joint_names (list, optional): Names of joints to update (overrides joint_idxs if not None).
            joint_idxs (list, optional): Indices of joints to update.

        Returns:
            None
        """
        # Update qpos if provided
        if q is not None:
            if joint_names is not None: # if 'joint_names' is not None, it override 'joint_idxs'
                joint_idxs = self.get_idxs_fwd(joint_names=joint_names)
            if joint_idxs is not None: 
                self.data.qpos[joint_idxs] = q
            else: self.data.qpos = q

        # Clip qpos to joint limits
        if clip_position:
            joints_to_clip = self.rev_pri_joint_names # only revolute and prismatic joints
            jidxs          = [self.model.joint(j).id for j in joints_to_clip]
            qidxs          = [int(self.model.jnt_qposadr[jidx]) for jidx in jidxs]  # qpos indices
            mins           = self.model.jnt_range[jidxs,0]
            maxs           = self.model.jnt_range[jidxs,1]
            self.data.qpos[qidxs] = np.clip(self.data.qpos[qidxs],mins,maxs)

        # Perform forward kinematics
        mujoco.mj_forward(self.model,self.data)

        # Update wall time
        self.increase_wall_time()

    def set_zero_qvel(self):
        """
        Set all joint velocities to zero and perform forward kinematics.
        """
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model,self.data)

    def compensate_gravity(self,root_body_names,on=True,exclude_free_joint=True):
        """
        Apply gravity compensation to the given root bodies.

        Parameters:
            root_body_names (list): List of root body names to apply gravity compensation.
            on (bool): If True, apply gravity compensation; if False, clear applied forces.
            exclude_free_joint (bool): If True, exclude free joints from gravity compensation.
        """

        # Always clear first so the result is not accumulated from previous calls
        self.data.qfrc_applied[:] = 0.0

        # Early return for OFF mode
        if not on:
            return
        jac = np.zeros((3,self.model.nv),dtype=np.float64) # (3, n_v) Jacobian for center of mass

        for root_body_name in root_body_names:
            subtree_id = self.model.body(root_body_name).id
            total_mass = float(self.model.body_subtreemass[subtree_id])

            mujoco.mj_jacSubtreeCom(
                self.model,
                self.data,
                jac,
                subtree_id,
            )

            f_grav = total_mass*self.model.opt.gravity
            qfrc_sub = -(jac.T@f_grav)

            if exclude_free_joint and (self.n_free_joint>0):
                free_dof_idxs = []
                for jidx in self.free_joint_idxs:
                    dofadr = int(self.model.jnt_dofadr[jidx])
                    free_dof_idxs.extend(range(dofadr,dofadr+6))
                qfrc_sub[np.array(free_dof_idxs,dtype=np.int32)] = 0.0

            self.data.qfrc_applied[:] += qfrc_sub       

    def get_idxs_fwd(self,joint_names):
        """
        Get the indices of joints used for forward kinematics based on joint names.
        
        Parameters:
            joint_names (list): List of joint names.
        
        Returns:
            list: Indices corresponding to the joints.
        """
        return [self.model.joint(jname).qposadr[0] for jname in joint_names]
    
    def get_idxs_step(self,joint_names):
        """
        Get the indices used for applying control during simulation steps based on joint names.
        
        Parameters:
            joint_names (list): List of joint names.
        
        Returns:
            list: Control indices.
        """
        return [self.ctrl_qpos_names.index(jname) for jname in joint_names]

    def get_idxs_jac(self,joint_names):
        """
        Get DOF indices (Jacobian column indices) for given joint names.

        Parameters:
            joint_names (list): List of joint names.
        """
        idxs=[]
        for jname in joint_names:
            j=self.model.joint(jname)
            jtype=int(j.type[0])
            dofadr=int(j.dofadr[0])

            if jtype==mujoco.mjtJoint.mjJNT_FREE:
                dof_count=6
            elif jtype==mujoco.mjtJoint.mjJNT_BALL:
                dof_count=3
            else:
                # HINGE / SLIDE
                dof_count=1

            idxs.extend(range(dofadr,dofadr+dof_count))
        return idxs
        
    def get_sim_time(self,init_flag=False):
        """
        Get the simulation time elapsed since initialization.

        Parameters:
            init_flag (bool): If True, resets the reference time to the current time.

        Returns:
            float: Simulation time in seconds.
        """
        if init_flag:
            self.init_sim_time = self.data.time
        elapsed_time = self.data.time - self.init_sim_time
        return elapsed_time
    
    def reset_sim_time(self):
        """
        Reset the simulation time reference to the current data.time.
        """
        self.init_sim_time = self.data.time
        
    def reset_wall_time(self):
        """
        Reset the wall-clock time reference.
        """
        current_wall_time = time.time()
        self.init_wall_time = current_wall_time
        self.accum_wall_time = 0.0
        self.last_wall_update = current_wall_time

    def increase_wall_time(self,step_flag=True):
        """
        Update the accumulated wall-clock time.

        Parameters:
            step_flag (bool): If True, add elapsed time to the accumulator.

        Returns:
            None
        """
        current_wall_time = time.time()
        if step_flag:
            # Increment wall time only when 'step_flag' is True
            self.accum_wall_time += current_wall_time - self.last_wall_update
        self.last_wall_update = current_wall_time        
        
    def get_wall_time(self,init_flag=False):
        """
        Get the wall-clock time elapsed since the last reset.
        This time is update only when 'step()' is called with 'step_flag=True' 
        or 'forward()' is called.

        Parameters:
            init_flag (bool): If True, reset the wall-time reference.

        Returns:
            float: Elapsed wall time in seconds.
        """
        if init_flag:
            self.accum_wall_time = 0.0
            self.last_wall_update = time.time()
        return self.accum_wall_time

    def get_key_pressed_list(self):
        """
        Get the list of keys that have been pressed.
        
        Returns:
            list: List of pressed keys.
        """
        if self.viewer is None: # if viewer is not initialized 
            return []
        
        # Return 
        return list(self.viewer._key_pressed_set)
    
    def grab_current_rgbd_imgs(self):
        """
        Capture the current RGB and depth images from the viewer.

        Returns:
            rgb_img (np.array): Captured RGB image (H x W x 3).
            depth_img (np.array): Captured depth image (H x W).
        """
        rgb_img   = np.zeros((self.viewer.viewport.height,self.viewer.viewport.width,3),dtype=np.uint8)
        depth_img = np.zeros((self.viewer.viewport.height,self.viewer.viewport.width,1), dtype=np.float32)
        mujoco.mjr_readPixels(rgb_img,depth_img,self.viewer.viewport,self.viewer.ctx)
        rgb_img,depth_img = np.flipud(rgb_img),np.flipud(depth_img) # flip up-down

        # Rescale depth image
        extent = self.model.stat.extent
        near   = self.model.vis.map.znear * extent
        far    = self.model.vis.map.zfar * extent
        scaled_depth_img = near / (1 - depth_img * (1 - near / far))
        depth_img = scaled_depth_img.squeeze()
        return rgb_img,depth_img
    
    def get_qpos(self,joint_names=None,ctrl_names=None,flatten=True):
        """
        Get joint position(s) (qpos) from the simulation.

        Parameters:
            joint_names (list[str] | None):
                Joint names to query.
            ctrl_names (list[str] | None):
                Actuator (control) names whose attached joint qpos will be returned.
            flatten (bool):
                - True : return concatenated 1D ndarray
                - False: return per-joint qpos (ndarray, dtype=object)

        Returns:
            np.ndarray:
                - no names provided -> full qpos (shape (nq,))
                - names provided:
                    - flatten False -> shape (N,), dtype=object
                    - flatten True  -> shape (sum qpos lengths,)
        """
        if joint_names is not None and ctrl_names is not None:
            raise ValueError("Provide only one of joint_names or ctrl_names.")

        # full qpos
        if joint_names is None and ctrl_names is None:
            return self.data.qpos.copy()

        # resolve ctrl_names -> joint_names
        if ctrl_names is not None:
            joint_names = []
            for cname in ctrl_names:
                act = self.model.actuator(cname)
                joint_id = int(act.trnid[0])
                if joint_id < 0:
                    raise ValueError(f"Actuator '{cname}' is not attached to a joint.")
                joint_names.append(self.joint_names[joint_id])

        # collect qpos slices for each joint
        q_list = []
        for jname in joint_names:
            j = self.model.joint(jname)
            addr = int(j.qposadr[0])
            L = int(len(j.qpos0))  # FREE=7, BALL=4, HINGE/SLIDE=1
            q_list.append(self.data.qpos[addr:addr+L].copy())

        # return
        if flatten:
            return np.concatenate(q_list) if q_list else np.array([])
        else:
            return np.array(q_list,dtype=object)

    def get_ctrl_qpos(self,ctrl_names=None):
        """
        Get joint positions attached to actuators.

        Assumes each actuator is attached to a single revolute or prismatic joint,
        so each control corresponds to exactly one qpos scalar.

        Parameters:
            ctrl_names (list | None): Actuator names. If None, use all actuators.

        Returns:
            np.ndarray: Joint positions in the same order as ctrl_names.
        """
        if ctrl_names is None:
            ctrl_names = self.ctrl_names

        ctrl_idxs = get_idxs(self.ctrl_names,ctrl_names)

        # Reject non-joint transmissions such as tendon actuators
        unsupported_ctrls = []
        for ctrl_idx in ctrl_idxs:
            if self.ctrl_types[ctrl_idx] != 'JOINT':
                unsupported_ctrls.append(self.ctrl_names[ctrl_idx])

        if len(unsupported_ctrls) > 0:
            raise ValueError(
                "[get_ctrl_qpos] Unsupported control type for ctrl_names:%s. "
                "Only JOINT transmission is supported."%(unsupported_ctrls)
            )

        qpos_idxs = np.array(self.ctrl_qpos_idxs,dtype=np.int32)[ctrl_idxs]
        return self.data.qpos[qpos_idxs].copy()

    def get_p(self,name,type):
        """
        Get the position of a specified object in the simulation.

        Parameters:
            name (str): Name of the object.
            type (str): Type of the object ('body', 'geom', 'site', 'camera', 'joint', 'sensor').

        Returns:
            np.array: 3-element position vector.
        """
        # Validate type
        assert type in ['body','geom','site','camera','joint','sensor'], "Unknown type: %s"%(type)

        # Get position based on type
        if type == 'body':
            obj_id = self.model.body(name).id
            p = self.data.xpos[obj_id]
        elif type == 'geom':
            obj_id = self.model.geom(name).id
            p = self.data.geom_xpos[obj_id]
        elif type == 'site':
            obj_id = self.model.site(name).id
            p = self.data.site_xpos[obj_id]
        elif type == 'camera':
            obj_id = self.model.cam(name).id
            p = self.data.cam_xpos[obj_id]
        elif type == 'joint':
            obj_id = self.model.joint(name).id
            body_id = self.model.joint(obj_id).bodyid[0]
            p = self.data.xpos[body_id]
        elif type == 'sensor':
            sensor_id = self.model.sensor(name).id # get sensor ID
            sensor_objtype = self.model.sensor_objtype[sensor_id] # get attached object type (i.e., site)
            sensor_objid = self.model.sensor_objid[sensor_id] # get attached object ID
            obj_name = mujoco.mj_id2name(self.model,sensor_objtype,sensor_objid) # get the site name
            if sensor_objtype == mujoco.mjtObj.mjOBJ_JOINT:
                body_id = self.model.joint(obj_name).bodyid[0]
                p = self.data.xpos[body_id] # get the position of the body attached to the joint
            elif sensor_objtype == mujoco.mjtObj.mjOBJ_SITE:
                p = self.data.site(obj_name).xpos # get the position of the site
            else:
                raise ValueError("Unknown sensor object type:[%d]"%(sensor_objtype))
        else:
            raise ValueError("Unknown type: %s"%(type))
        
        # Return
        return p.copy()
    
    def get_R(self,name,type):
        """
        Get the rotation matrix of a specified object in the simulation.

        Parameters:
            name (str): Name of the object.
            type (str): Type of the object ('body', 'geom', 'site', 'camera', 'joint', 'sensor').

        Returns:
            np.array: 3x3 rotation matrix.
        """
        # Validate type
        assert type in ['body','geom','site','camera','joint','sensor'], "Unknown type: %s"%(type)

        # Get rotation based on type
        if type == 'body':
            obj_id = self.model.body(name).id
            R = self.data.xmat[obj_id].reshape(3,3)
        elif type == 'geom':
            obj_id = self.model.geom(name).id
            R = self.data.geom_xmat[obj_id].reshape(3,3)
        elif type == 'site':
            obj_id = self.model.site(name).id
            R = self.data.site_xmat[obj_id].reshape(3,3)
        elif type == 'camera':
            obj_id = self.model.cam(name).id
            R = self.data.cam_xmat[obj_id].reshape(3,3)
        elif type == 'joint':
            obj_id = self.model.joint(name).id
            body_id = self.model.joint(obj_id).bodyid[0]
            R = self.data.xmat[body_id].reshape(3,3)
        elif type == 'sensor':
            sensor_id = self.model.sensor(name).id # get sensor ID
            sensor_objtype = self.model.sensor_objtype[sensor_id] # get attached object type (i.e., site)
            sensor_objid = self.model.sensor_objid[sensor_id] # get attached object ID
            obj_name = mujoco.mj_id2name(self.model,sensor_objtype,sensor_objid) # get the site name
            if sensor_objtype == mujoco.mjtObj.mjOBJ_JOINT:
                body_id = self.model.joint(obj_name).bodyid[0]
                R = self.data.xmat[body_id].reshape(3,3) # get the rotation of the body attached to the joint
            elif sensor_objtype == mujoco.mjtObj.mjOBJ_SITE:
                R = self.data.site(obj_name).xmat.reshape(3,3) # get the rotation of the site
            else:
                raise ValueError("Unknown sensor object type:[%d]"%(sensor_objtype))
        else:
            raise ValueError("Unknown type: %s"%(type))
        
        # Return
        return R.copy()
    
    def get_T(self,name,type):
        """
        Get the transformation matrix of a specified object in the simulation.

        Parameters:
            name (str): Name of the object.
            type (str): Type of the object ('body', 'geom', 'site', 'camera', 'joint', 'sensor').

        Returns:
            np.array: 4x4 transformation matrix.
        """
        p = self.get_p(name,type)
        R = self.get_R(name,type)
        T = pr2t(p,R)
        return T
    
    def get_J_site(self,site_name,return_type='all'):
        """
        Compute the Jacobian matrices for the specified site.
        
        Parameters:
            site_name (str): Name of the site.
            return_type (str): Type of Jacobian to return ('all', 'pos', 'rot', 'full').
        
        Returns:
            tuple: (J_p, J_R, J_full)
        """
        J_p = np.zeros((3,self.n_v)) # nv: nDoF
        J_R = np.zeros((3,self.n_v))
        mujoco.mj_jacSite(self.model,self.data,J_p,J_R,self.data.site(site_name).id)
        J_full = np.array(np.vstack([J_p,J_R]))

        if return_type=='all':
            return J_p,J_R,J_full    
        elif return_type=='pos':
            return J_p
        elif return_type=='rot':
            return J_R
        elif return_type=='full':
            return J_full
        else:
            print ("[get_J_site] return_type:[%s] is not supported!"%(return_type))
            return None
        
    def _set_camera_local_pose(
            self,
            cam_name,
            p_world = None,
            R_world = None,
        ):
        """
        Set a camera pose using world-frame position/rotation.

        Internally MuJoCo stores camera pose relative to the parent body,
        so this helper converts the given world pose into the local frame
        of the body that owns the camera.

        Parameters:
            cam_name (str): Camera name.
            p_world (np.array | None): Target world position.
            R_world (np.array | None): Target world rotation matrix.
        """
        cam_id     = self.model.cam(cam_name).id
        parent_bid = int(self.model.cam_bodyid[cam_id])

        if parent_bid == 0:
            p_parent = np.zeros(3,dtype=np.float64)
            R_parent = np.eye(3,dtype=np.float64)
        else:
            p_parent = self.data.xpos[parent_bid].copy()
            R_parent = self.data.xmat[parent_bid].reshape(3,3).copy()

        if p_world is not None:
            p_world = np.asarray(p_world,dtype=np.float64).reshape(3,)
            p_local = R_parent.T @ (p_world-p_parent)
            self.model.cam_pos[cam_id] = p_local

        if R_world is not None:
            R_world = np.asarray(R_world,dtype=np.float64).reshape(3,3)
            R_local = R_parent.T @ R_world
            self.model.cam_quat[cam_id] = r2quat(R_local)

    """
    Set position, rotation, and transformation matrix of an object
    ('base', 'base_body', 'body', 'mocap', 'camera')
        - set_p: set position
        - set_R: set rotation
        - set_T: set transformation
    """
    def set_p(self,name,type,p,forward=True):
        """
        Set the position of a specified object in the simulation.

        Parameters:
            name (str): Name of the object.
            type (str): Type of the object ('base', 'base_body', 'body', 'mocap', 'camera').
            p (np.array): 3-element position vector.
            forward (bool): Whether to perform forward kinematics after setting the position.
        """
        p = np.asarray(p).flatten()
        if type == 'base' or type == 'base_body': # set position of the base body
            jntadr  = self.model.body(name).jntadr[0]
            qposadr = self.model.jnt_qposadr[jntadr]
            self.data.qpos[qposadr:qposadr+3] = p
        elif type == 'body': # set position of a body
            self.model.body(name).pos = p
        elif type == 'mocap': # set position of a mocap body
            mocap_idx = self.model.body_mocapid[self.body_names.index(name)]
            self.data.mocap_pos[mocap_idx] = p
        elif type == 'camera': # set world position of a camera
            self._set_camera_local_pose(cam_name=name,p_world=p,R_world=None)
        else:
            raise ValueError("Cannot set position for type:[%s]"%(type))
        
        # Forward kinematics
        if forward:
            mujoco.mj_forward(self.model,self.data)

    def set_R(self,name,type,R,forward=True):
        """
        Set the rotation of a specified object in the simulation.

        Parameters:
            name (str): Name of the object.
            type (str): Type of the object ('base', 'base_body', 'body', 'mocap', 'camera').
            R (np.array): 3x3 rotation matrix.
            forward (bool): Whether to perform forward kinematics after setting the rotation.
        """
        R = np.asarray(R).reshape(3,3)
        if type == 'base' or type == 'base_body': # set rotation of the base body
            jntadr  = self.model.body(name).jntadr[0]
            qposadr = self.model.jnt_qposadr[jntadr]
            self.data.qpos[qposadr+3:qposadr+7] = r2quat(R)
        elif type == 'body': # set rotation of a body
            self.model.body(name).quat = r2quat(R)
        elif type == 'mocap': # set rotation of a mocap body
            mocap_idx = self.model.body_mocapid[self.body_names.index(name)]
            self.data.mocap_quat[mocap_idx] = r2quat(R)
        elif type == 'camera': # set world rotation of a camera
            self._set_camera_local_pose(cam_name=name,p_world=None,R_world=R)
        else:
            raise ValueError("Cannot set rotation for type:[%s]"%(type))
        
        # Forward kinematics
        if forward:
            mujoco.mj_forward(self.model,self.data)

    def set_T(self,name,type,T,forward=True):
        """
        Set the transformation matrix of a specified object in the simulation.

        Parameters:
            name (str): Name of the object.
            type (str): Type of the object ('base', 'base_body', 'body', 'mocap', 'camera').
            T (np.array): 4x4 transformation matrix.
            forward (bool): Whether to perform forward kinematics after setting the transformation.
        """
        p = t2p(T)
        R = t2r(T)
        self.set_p(name,type,p,forward=False)
        self.set_R(name,type,R,forward=False)

        # Forward kinematics
        if forward:
            mujoco.mj_forward(self.model,self.data)

    def plot_T(
            self,
            T           = None,
            p           = np.array([0,0,0]),
            R           = np.eye(3),
            plot_axis   = True,
            axis_len    = 1.0,
            axis_width  = 0.005,
            axis_rgba   = None,
            axis_alpha  = None,
            plot_sphere = False,
            sphere_r    = 0.05,
            sphere_rgba = [1,0,0,0.5],
            label       = None,
            print_xyz   = False,
        ):
        """
        Plot coordinate axes (and optionally a sphere and label) at the given pose.
        
        Parameters:
            T (np.array): 4x4 transformation matrix. If provided, it overrides p and R.
            p (np.array): Position.
            R (np.array): Rotation matrix. 
            plot_axis (bool): Whether to plot the coordinate axes.
            axis_len (float): Length of each axis.
            axis_width (float): Thickness of the axes.
            axis_rgba (list): RGBA colors for the axes.
            plot_sphere (bool): Whether to plot a sphere marker at p.
            sphere_r (float): Radius of the sphere.
            sphere_rgba (list): RGBA color of the sphere.
            label (str): Optional text label.
            print_xyz (bool): Whether to print coordinate info.
        """
        if T is not None: # if T is not None, it overrides p and R
            p = t2p(T)
            R = t2r(T)
        # Add axis markers
        if plot_axis:
            if axis_alpha is None: axis_alpha = 0.9
            if axis_rgba is None:
                rgba_x = [1.0,0.0,0.0,axis_alpha]
                rgba_y = [0.0,1.0,0.0,axis_alpha]
                rgba_z = [0.0,0.0,1.0,axis_alpha]
            else:
                rgba_x = axis_rgba
                rgba_y = axis_rgba
                rgba_z = axis_rgba
            R_x = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([1,0,0]))
            p_x = p+R_x[:,2]*axis_len/2
            if print_xyz: axis_label = 'X-axis'
            else: axis_label = ''
            self.viewer.add_marker(
                pos   = p_x,
                type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size  = [axis_width,axis_width,axis_len/2],
                mat   = R_x,
                rgba  = rgba_x,
                label = axis_label,
            )
            R_y = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([0,1,0]))
            p_y = p + R_y[:,2]*axis_len/2
            if print_xyz: axis_label = 'Y-axis'
            else: axis_label = ''
            self.viewer.add_marker(
                pos   = p_y,
                type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size  = [axis_width,axis_width,axis_len/2],
                mat   = R_y,
                rgba  = rgba_y,
                label = axis_label,
            )
            R_z = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([0,0,1]))
            p_z = p + R_z[:,2]*axis_len/2
            if print_xyz: axis_label = 'Z-axis'
            else: axis_label = ''
            self.viewer.add_marker(
                pos   = p_z,
                type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size  = [axis_width,axis_width,axis_len/2],
                mat   = R_z,
                rgba  = rgba_z,
                label = axis_label,
            )
        # Add sphere marker
        if plot_sphere:
            self.viewer.add_marker(
                pos   = p,
                size  = [sphere_r,sphere_r,sphere_r],
                rgba  = sphere_rgba,
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = '')
        # Add label
        if label is not None:
            self.viewer.add_marker(
                pos   = p,
                size  = [0.0001,0.0001,0.0001],
                rgba  = [1,1,1,0.01],
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = label,
            )

    def plot_global_coordinate_axes(
            self,
            p_offset  = np.array([0,0,0]),
            axis_len  = 1.0,
            r         = 0.001,
            alpha     = 0.5,
            print_xyz = False,
        ):
        """
        Plot the global coordinate axes in the viewer.

        Parameters:
            p_offset (np.array): Position offset for the origin of the axes.
            axis_len (float): Length of each axis.
            r (float): Thickness of the axes.
            alpha (float): Transparency of the axes.
            print_xyz (bool): Whether to print coordinate info.
        """
        self.plot_T(
            p          = np.array(p_offset),
            axis_len   = axis_len,
            axis_width = r,
            axis_rgba  = None,
            axis_alpha = alpha,
            print_xyz  = print_xyz,
        )

    def plot_sphere(self,p,r=0.1,rgba=[1,1,1,1],label='',show_xyz=False):
        """
        Plot a sphere marker at the specified position.
        
        Parameters:
            p (np.array): Position (2D or 3D).
            r (float): Radius of the sphere.
            rgba (list): RGBA color.
            label (str): Optional label.
        """
        if p is None: return
        p = np.asarray(p)
        if show_xyz:
            label = '(%.2f, %.2f, %.2f)'%(p[0],p[1],p[2])
        if len(p) == 2: # only x and y are given (pad z=0)
            self.viewer.add_marker(
                pos   = np.append(p,[0]),
                size  = [r,r,r],
                rgba  = rgba,
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = label,
            )
        elif len(p) == 3:
            self.viewer.add_marker(
                pos   = p,
                size  = [r,r,r],
                rgba  = rgba,
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = label,
            )

    def plot_spheres(self,p_list,r=0.1,rgba=[1,1,1,1],label=''):
        """
        Plot multiple spheres at the positions given in p_list.
        
        Parameters:
            p_list (list of np.array): List of positions.
            r (float): Radius for each sphere.
            rgba (list): RGBA color.
            label (str): Optional label for each sphere.
        """
        for p in p_list:
            self.plot_sphere(p=p,r=r,rgba=rgba,label=label)

    def plot_box(
            self,
            T          = None,
            p          = np.array([0,0,0]),
            R          = np.eye(3),
            xlen       = 0.1,
            ylen       = 0.1,
            zlen       = 0.1,
            xyz_len    = None,
            rgba       = [0.5,0.5,0.5,0.5],
            label      = '',
            plot_axes  = False,
            axis_len   = 0.05,
            axis_width = 0.002,
            axis_alpha = 0.9,
        ):
        """
        Plot a box marker at the specified pose.
        
        Parameters:
            T (np.array): 4x4 transformation matrix. If provided, it overrides p and R.
            p (np.array): Position.
            R (np.array): Orientation.
            xlen (float): Length along the x-axis.
            ylen (float): Length along the y-axis.
            zlen (float): Length along the z-axis.
            xyz_len (np.array): 3-element array specifying [xlen, ylen, zlen]. If provided, it overrides xlen, ylen, zlen.
            rgba (list): RGBA color.
            label (str): Optional label.
            plot_axes (bool): Whether to plot coordinate axes.
            axis_len (float): Length of the axes.
            axis_width (float): Thickness of the axes.
            axis_alpha (float): Transparency of the axes.
        """
        if T is not None: # if T is given, it overrides p and R
            p = t2p(T)
            R = t2r(T)
        pos = np.asarray(p)
        mat = np.asarray(R, dtype=np.float64).reshape(9,)

        if xyz_len is not None: # if xyz_len is given, it overrides xlen, ylen, zlen
            xyz_len = np.asarray(xyz_len)
            xlen = xyz_len[0]
            ylen = xyz_len[1]
            zlen = xyz_len[2]
            
        self.viewer.add_marker(
            pos   = pos,
            mat   = mat,
            type  = mujoco.mjtGeom.mjGEOM_BOX,
            size  = [xlen/2,ylen/2,zlen/2],
            rgba  = rgba,
            label = label,
        )

        if plot_axes:
            self.plot_T(
                p          = p,
                R          = R,
                axis_len   = axis_len,
                axis_width = axis_width,
                axis_alpha = axis_alpha,
            )

    def plot_arrow_fr2to(
            self,
            p_fr,
            p_to,
            r          = 1.0,
            rgba       = [0.5,0.5,0.5,0.5],
            label      = '',
            label_to   = None,
            plot_axes  = False,
            axis_len   = 0.05,
            axis_width = 0.002,
            axis_alpha = 0.9
        ):
        """
        Plot an arrow from point p_fr to point p_to.
        
        Parameters:
            p_fr (np.array): Starting point of the arrow.
            p_to (np.array): Ending point of the arrow.
            r (float): Radius of the arrow shaft.
            rgba (list): RGBA color.
            label (str): Optional label for the starting point.
            label_to (str): Optional label for the ending point.
            plot_axes  (bool): Whether to plot coordinate axes at the arrow's start and end points.
            axis_len   (float): Length of the coordinate axes.
            axis_width (float): Width of the coordinate axes.
            axis_alpha (float): Transparency of the coordinate axes.
        """
        # Ensure p_fr and p_to are numpy arrays
        p_fr = np.asarray(p_fr)
        p_to = np.asarray(p_to)
        R_fr2to = get_R_from_twopoints(p_fr=p_fr,p_to=p_to)
        self.viewer.add_marker(
            pos   = p_fr,
            mat   = R_fr2to,
            type  = mujoco.mjtGeom.mjGEOM_ARROW,
            size  = [r,r,np.linalg.norm(p_to-p_fr)*2],
            rgba  = rgba,
            label = label,
        )
        if label_to is not None:
            self.viewer.add_marker(
                pos   = p_to,
                size  = [0.0001,0.0001,0.0001],
                rgba  = [1,1,1,0.0],
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = label_to,
            )

        if plot_axes:
            self.plot_T(
                p          = p_fr,
                R          = R_fr2to,
                axis_len   = axis_len,
                axis_width = axis_width,
                axis_alpha = axis_alpha,
            )

    def plot_line_fr2to(
            self,
            p_fr,
            p_to,
            rgba  = [0.5,0.5,0.5,0.5],
            label = '',
        ):
        """
        Plot a line connecting two points.
        
        Parameters:
            p_fr (np.array): Starting point.
            p_to (np.array): Ending point.
            rgba (list): RGBA color.
            label (str): Optional label.
        """
        # Ensure p_fr and p_to are numpy arrays
        p_fr = np.asarray(p_fr)
        p_to = np.asarray(p_to)
        if np.linalg.norm(p_to-p_fr) < 1e-6: # if the two points are too close, do not plot
            return
        R_fr2to = get_R_from_twopoints(p_fr=p_fr,p_to=p_to)
        self.viewer.add_marker(
            pos   = p_fr,
            mat   = R_fr2to,
            type  = mujoco.mjtGeom.mjGEOM_LINE,
            size  = np.linalg.norm(p_to-p_fr),
            rgba  = rgba,
            label = label,
        )

    def plot_cylinder_fr2to(
            self,
            p_fr,
            p_to,
            r        = 0.01,
            rgba     = [0.5,0.5,0.5,0.5],
            label    = '',
            label_to = None
        ):
        """
        Plot a cylinder marker between two points.
        
        Parameters:
            p_fr (np.array): Starting point.
            p_to (np.array): Ending point.
            r (float): Cylinder radius.
            rgba (list): RGBA color.
            label (str): Optional label.
            label_to (str): Optional label for the ending point.
        """
        # Ensure p_fr and p_to are numpy arrays
        p_fr = np.asarray(p_fr)
        p_to = np.asarray(p_to)
        if np.linalg.norm(p_to-p_fr) < 1e-6: # if the two points are too close, do not plot
            return
        R_fr2to = get_R_from_twopoints(p_fr=p_fr,p_to=p_to)
        self.viewer.add_marker(
            pos   = (p_fr+p_to)/2,
            mat   = R_fr2to,
            type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
            size  = [r,r,np.linalg.norm(p_to-p_fr)/2],
            rgba  = rgba,
            label = label,
        )
        if label_to is not None:
            self.viewer.add_marker(
                pos   = p_to,
                size  = [0.0001,0.0001,0.0001],
                rgba  = [1,1,1,0.0],
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = label_to,
            )

    def plot_sites(self,site_names=None,show_names=False):
        """
        Plot spheres at the positions of specified sites.

        Parameters:
            site_names (list): List of site names to plot. If None, all sites are plotted.
            show_names (bool): Whether to display site names as labels.
        """
        if site_names is None: # if site_names is not provided, plot all sites
            site_names = self.site_names
        for site_name in site_names: # iterate through all site names
            if site_name: # if site_name is valid
                p    = self.get_p(site_name,'site')
                R    = self.get_R(site_name,'site')
                size = self.model.site(site_name).size
                type = self.model.site(site_name).type
                rgba = self.model.site(site_name).rgba
                if show_names: label = site_name
                else: label = ''

                if type == mujoco.mjtGeom.mjGEOM_SPHERE:
                    r = size[0]
                    self.plot_sphere(p,r,rgba,label)
                elif type == mujoco.mjtGeom.mjGEOM_BOX:
                    xlen,ylen,zlen = size[0]*2,size[1]*2,size[2]*2
                    self.plot_box(
                        p        = p,
                        R        = R,
                        xlen     = xlen,
                        ylen     = ylen,
                        zlen     = zlen,
                        rgba     = rgba,
                        label    = label,
                    )
                else:
                    print("Site type [%d] not supported for plotting."%(type))

    def get_contact_info(self,forward=True):
        """
        Get detailed contact information from the simulation.

        Parameters:
            forward (bool): If True, perform a forward simulation step to update contact data.

        Returns:
            dict: A dictionary containing contact information, including:
                - 'n_contact': Number of contacts.
                - 'contact_list': List of contact objects.
                - 'p_contact_list': List of contact positions.
                - 'geom1_idx_list': List of first geometry indices.
                - 'geom2_idx_list': List of second geometry indices.
                - 'body1_idx_list': List of first body indices.
                - 'body2_idx_list': List of second body indices.
                - 'R_frame_list': List of contact frame rotation matrices.
                - 'f_world_list': List of contact forces in world coordinates.
                - 'norm_dir_list': List of contact normal directions.
                - 'min_contact_dist': Minimum contact distance.
        """
        # Perform a forward simulation step to ensure contact data is up-to-date
        if forward:
            self.forward()
        
        # Initialize lists to store contact information
        n_contact,contact_list,p_contact_list = 0,[],[]
        geom1_idx_list,geom2_idx_list = [],[]
        body1_idx_list,body2_idx_list = [],[]
        R_frame_list,f_world_list,norm_dir_list = [],[],[]
        min_contact_dist = np.inf

        # Iterate through all contacts
        for c_idx in range(self.data.ncon): # iterate through all contacts
            contact = self.data.contact[c_idx]

            # Contact position and frame orientation
            p_contact = np.array(contact.pos,dtype=np.float64) # (3,)
            R_frame = np.array(contact.frame,dtype=np.float64).reshape(3,3) # (3 x 3)
            
            # Contact force in local coordinates
            f_local = np.zeros(6,dtype=np.float64) # (6,)
            mujoco.mj_contactForce(self.model,self.data,c_idx,f_local) # contact force

            # Contact force in world coordinates
            f_world = R_frame.T@f_local[:3] # (3,)
            norm_dir = R_frame[0,:] # contact normal direction (3,)

            # Contacting geoms
            geom1_idx,geom2_idx = contact.geom1,contact.geom2
            if (geom1_idx<0) or (geom2_idx<0): continue # skip invalid contacts

            # Append contact related information
            n_contact = n_contact + 1
            body1_idx,body2_idx = self.model.geom_bodyid[geom1_idx],self.model.geom_bodyid[geom2_idx]
            contact_list.append(contact)
            p_contact_list.append(p_contact)
            geom1_idx_list.append(geom1_idx)
            geom2_idx_list.append(geom2_idx)
            body1_idx_list.append(body1_idx)
            body2_idx_list.append(body2_idx)
            R_frame_list.append(R_frame)
            f_world_list.append(f_world)
            norm_dir_list.append(norm_dir)

            # Min contact distance
            if contact.dist < min_contact_dist:
                min_contact_dist = contact.dist

        # Return
        contact_info = {
            'n_contact': n_contact,
            'contact_list': contact_list,
            'p_contact_list': p_contact_list,
            'geom1_idx_list': geom1_idx_list,
            'geom2_idx_list': geom2_idx_list,
            'body1_idx_list': body1_idx_list,
            'body2_idx_list': body2_idx_list,
            'R_frame_list': R_frame_list,
            'f_world_list': f_world_list,
            'norm_dir_list': norm_dir_list,
            'min_contact_dist': min_contact_dist,
        }
        return contact_info
    
    def plot_contact_info(
            self,
            contact_info = None,
            plot_arrow   = True,
            cmap_arrow   = None, # "rainbow"
            arrow_len    = 0.05,
            arrow_r      = 0.005,
            arrow_rgba1  = (0.9,0.3,0,1),
            arrow_rgba2  = (0.9,0,0.3,1),
            arrow_alpha  = None,
            plot_contact = True,
            contact_rgba = (1,0,0,0.5),
            contact_r    = 0.02,
            contact_h    = 0.01,
            rate         = 1.0,
        ):
        """
        Visualize contact information in the viewer.

        Parameters:
            contact_info (dict): Contact information dictionary. If None, it will be obtained from get_contact_info().
            cmap_arrow (str): Colormap name for contact arrows.
            arrow_len (float): Length of the contact arrows.
            arrow_r (float): Radius of the contact arrows.
            arrow_rgba1 (tuple): RGBA color for the first contact arrow.
            arrow_rgba2 (tuple): RGBA color for the second contact arrow.
            arrow_alpha (float or None): Transparency for contact arrows.
                If None, 0.8 is used. If specified, this alpha is always applied
                to arrows regardless of cmap_arrow or arrow_rgba1/2 alpha values.
            plot_contact (bool): Whether to plot cylinders at contact points.
            contact_rgba (tuple): RGBA color for the contact cylinders.
            contact_r (float): Radius of the contact cylinders.
            contact_h (float): Height of the contact cylinders.
            rate (float): Scaling factor for arrow length and contact size.
        """
        # Default arrow alpha
        if arrow_alpha is None:
            arrow_alpha = 0.8

        # Get contact information
        if contact_info is None:
            contact_info = self.get_contact_info()

        n_contact      = contact_info["n_contact"]
        p_contact_list = contact_info["p_contact_list"]
        geom1_idx_list = contact_info["geom1_idx_list"]
        geom2_idx_list = contact_info["geom2_idx_list"]
        body1_idx_list = contact_info["body1_idx_list"]
        body2_idx_list = contact_info["body2_idx_list"]
        norm_dir_list  = contact_info["norm_dir_list"]
        R_frame_list   = contact_info["R_frame_list"]

        if cmap_arrow is not None:
            colors_arrow = get_colors(
                n_color   = n_contact*2,
                cmap_name = cmap_arrow,
                alpha     = arrow_alpha,
            )

        for c_idx in range(n_contact):
            p_contact = p_contact_list[c_idx]
            norm_dir  = norm_dir_list[c_idx] # normal direction (3,)
            R_frame   = R_frame_list[c_idx]

            # Plot contact point
            if plot_contact:
                frame_norm = R_frame[0,:]
                norm_sq    = frame_norm[0]*frame_norm[0] + frame_norm[1]*frame_norm[1] + frame_norm[2]*frame_norm[2]
                if norm_sq > 0.0:
                    inv_norm   = 1.0 / np.sqrt(norm_sq)
                    frame_norm = frame_norm * inv_norm
                    if (frame_norm[0]*norm_dir[0] + frame_norm[1]*norm_dir[1] + frame_norm[2]*norm_dir[2]) < 0.0:
                        frame_norm = -frame_norm
                    norm_dir = frame_norm

                self.plot_cylinder_fr2to(
                    p_fr = p_contact - 0.5*contact_h*norm_dir*rate,
                    p_to = p_contact + 0.5*contact_h*norm_dir*rate,
                    r    = contact_r*rate,
                    rgba = contact_rgba,
                )

            # Plot contacting arrows
            geom1_idx,geom2_idx = geom1_idx_list[c_idx],geom2_idx_list[c_idx]
            body1_idx,body2_idx = body1_idx_list[c_idx],body2_idx_list[c_idx]
            body1_name,body2_name = self.body_names[body1_idx],self.body_names[body2_idx]

            if cmap_arrow is not None:
                rgba_arrow1 = colors_arrow[2*c_idx]
                rgba_arrow2 = colors_arrow[2*c_idx+1]
                rgba_arrow1 = (rgba_arrow1[0],rgba_arrow1[1],rgba_arrow1[2],arrow_alpha)
                rgba_arrow2 = (rgba_arrow2[0],rgba_arrow2[1],rgba_arrow2[2],arrow_alpha)
            else:
                rgba_arrow1 = (arrow_rgba1[0],arrow_rgba1[1],arrow_rgba1[2],arrow_alpha)
                rgba_arrow2 = (arrow_rgba2[0],arrow_rgba2[1],arrow_rgba2[2],arrow_alpha)

            p1_to = p_contact - arrow_len*norm_dir*rate
            p2_to = p_contact + arrow_len*norm_dir*rate

            if plot_arrow:
                self.plot_arrow_fr2to(p_contact,p1_to,arrow_r*rate,rgba_arrow1)
                self.plot_arrow_fr2to(p_contact,p2_to,arrow_r*rate,rgba_arrow2)    

    def plot_links_between_bodies(
            self,
            root_body_name = 'world',
            pose_env       = None,
            r              = 0.002,
            rgba           = (0.0,0.0,0.0,0.5),
            plot_sphere    = False,
            r_sphere       = 0.01,
            rgba_sphere    = (0.0,0.0,0.0,0.5),
            eps            = 1e-9,
            exclude_list   = None,
        ):
        """
        Plot cylinders connecting parent-child bodies within the subtree rooted at root_body_name.

        Parameters:
            root_body_name (str): Name of the root body for the subtree.
            pose_env (MuJoCoParser | None): Environment to read body poses from.
                If None, read poses from self. This is useful for rendering a
                target or ghost posture in the current viewer while computing
                that posture in a separate kinematic environment.
            r (float): Radius of the cylinders.
            rgba (tuple): RGBA color of the cylinders.
            plot_sphere (bool): Whether to plot spheres at the parent body positions.
            r_sphere (float): Radius of the spheres.
            rgba_sphere (tuple): RGBA color of the spheres.
            eps (float): Minimum distance threshold to plot a cylinder.
            exclude_list (list | None): List of body names to exclude from plotting.
        """
        pose_env = self if pose_env is None else pose_env

        # Get subtree body names (exclude root by default)
        subtree_body_names=self.get_subtree_body_names(root_body_name=root_body_name,include_root=False)

        # Draw links: for each body in subtree, connect it to its parent (if parent exists)
        for body_name in subtree_body_names:
            if body_name is None: continue
            body_idx         = self.body_names.index(body_name)
            parent_body_name = self.parent_body_names[body_idx]

            # Skip world parent
            if parent_body_name is None:
                continue

            # Skip excluded bodies
            if exclude_list is not None and body_name in exclude_list:
                continue

            p_parent = pose_env.get_p(parent_body_name,'body')
            p_child  = pose_env.get_p(body_name,'body')

            if np.linalg.norm(p_child-p_parent)<eps:
                continue

            self.plot_cylinder_fr2to(p_fr=p_parent,p_to=p_child,r=r,rgba=rgba)

            if plot_sphere:
                self.plot_sphere(p=p_parent,r=r_sphere,rgba=rgba_sphere)
        
    def plot_rev_joint_axis(
            self,
            rev_joint_names = None,
            axis_len        = 0.1,
            axis_r          = 0.01,
            alpha           = 0.2,
            rate            = 1.0,
            print_name      = False,
            axis_colors     = None,
        ):
        """
        Plot the axis of revolute (hinge) joints as arrows in world coordinates.

        Parameters:
            rev_joint_names (list | None): Revolute joint names to plot. If None, use self.rev_joint_names.
            axis_len (float): Arrow length.
            axis_r (float): Arrow radius.
            alpha (float): Default alpha when axis_colors is None.
            rate (float): Global scale factor for axis_len and axis_r.
            print_name (bool): If True, show joint name labels.
            axis_colors (list | None): Optional list of RGBA colors per joint (same order as rev_joint_names).
        """
        if rev_joint_names is None:
            rev_joint_names=self.rev_joint_names

        idxs=get_idxs(self.rev_joint_names,rev_joint_names)
        rev_joint_idxs=self.rev_joint_idxs[idxs]

        for i,(jidx,jname) in enumerate(zip(rev_joint_idxs,rev_joint_names)):
            axis_joint=self.model.jnt_axis[jidx]

            body_id=int(self.model.joint(jname).bodyid[0])
            body_name=self.body_names[body_id]

            p_joint=self.get_p(body_name,'body')
            R_joint=self.get_R(body_name,'body')
            axis_world=R_joint@axis_joint

            if axis_colors is None:
                axis_rgba=np.append(np.eye(3)[:,int(np.argmax(np.abs(axis_joint)))],alpha)
            else:
                axis_rgba=axis_colors[i] if i<len(axis_colors) else axis_colors[-1]

            self.plot_arrow_fr2to(
                p_fr=p_joint,
                p_to=p_joint+rate*axis_len*axis_world,
                r=rate*axis_r,
                rgba=axis_rgba,
            )

            if print_name:
                self.viewer.add_marker(
                    pos=p_joint,
                    size=[0.0001,0.0001,0.0001],
                    rgba=[1,1,1,0.01],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    label=jname,
                )

    def plot_time(
            self,
            loc = 'bottom left',
            plot_sim_time  = True,
            plot_wall_time = True,
            plot_loop_freq = False,
            plot_time_diff = True,
        ):
        """
        Overlay the current simulation tick, simulation time, and wall-clock time on the viewer.
        
        Parameters:
            loc (str): Location of the text overlay. Options include 'top left', 'top right', 'bottom left', 'bottom right'.
            plot_sim_time (bool): Whether to display the current simulation time.
            plot_wall_time (bool): Whether to display the current wall-clock time.
            plot_loop_freq (bool): Whether to display the loop frequency (Hz).
            plot_time_diff (bool): Whether to display the difference between wall time and simulation time.
        
        Returns:
            None
        """
        if plot_sim_time:
            self.viewer_text_overlay(text1='sim time',text2='[%.2f]sec'%(self.get_sim_time()),loc=loc)
        if plot_wall_time:
            self.viewer_text_overlay(text1='wall time',text2='[%.2f]sec'%(self.get_wall_time()),loc=loc)
        if plot_loop_freq:
            self.viewer_text_overlay(text1="Loop freq",text2="[%.1f]Hz"%(self.loop_hz),loc=loc)
            self.viewer_text_overlay(text1="Loop freq (ema)",text2="[%.1f]Hz"%(self.loop_hz_ema),loc=loc)
        if plot_time_diff:
            time_diff = self.get_wall_time() - self.get_sim_time()
            self.viewer_text_overlay(text1="Time diff",text2="[%.2f]sec"%(time_diff),loc=loc)
    
    @staticmethod
    def eval_quartic(x,coef=np.zeros(5),x0=0.0,y0=0.0):
        """
        Evaluate a quartic polynomial function at a given point x.

        Parameters:
            x (float): The point at which to evaluate the polynomial.
            coef (list): Coefficients of the polynomial, ordered from constant term to highest degree.
            x0 (float): Offset for the x-coordinate.
        """
        x_bar = float(x)-x0
        return y0+coef[0]+coef[1]*x_bar+coef[2]*x_bar**2+coef[3]*x_bar**3+coef[4]*x_bar**4

    @staticmethod
    def eval_quartic_derivative(x,coef=np.zeros(5),x0=0.0,y0=0.0):
        """
        Evaluate the derivative of a quartic polynomial function at a given point x.

        Parameters:
            x (float): The point at which to evaluate the derivative.
            coef (list): Coefficients of the polynomial, ordered from constant term to highest degree.
            x0 (float): Offset for the x-coordinate.
        """
        x_bar = x-x0
        return coef[1]+2*coef[2]*x_bar+3*coef[3]*x_bar**2+4*coef[4]*x_bar**3
    
    def get_qrev_from_qactive(self,qactive):
        """
        Given active joint positions, compute the full set of joint positions
        including passive joints based on equality constraints.

        Parameters:
            qactive (list or np.array): Active joint positions.

        Returns:
            np.array: Full joint positions including passive joints.
        """
        # Initialize rev joint positions
        qrev = self.get_qpos(joint_names=self.rev_joint_names)

        # Set active revolute joint positions to rev joint positions
        active_rev_joint_names = self.active_rev_joint_names
        qactive = np.asarray(qactive,dtype=np.float64).reshape(-1,)
        if len(qactive) == len(self.active_joint_names):
            qactive_map = {j:qactive[idx] for idx,j in enumerate(self.active_joint_names)}
            qactive_rev = np.array([qactive_map[j] for j in active_rev_joint_names],dtype=np.float64)
        else:
            qactive_rev = qactive
        qrev[get_idxs(self.rev_joint_names,active_rev_joint_names)] = qactive_rev

        # Update passive joint values based on equality constraints
        for eq_idx in range(self.n_eq): # for each joint equality constraints
            if self.model.eq_type[eq_idx] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            passive_joint = self.model.joint(self.model.eq_obj1id[eq_idx]).name
            active_joint  = self.model.joint(self.model.eq_obj2id[eq_idx]).name
            coef = self.model.eq_data[eq_idx,:5]
            x0   = self.model.joint(active_joint).qpos0[0]
            y0   = self.model.joint(passive_joint).qpos0[0]
            x    = qactive_rev[active_rev_joint_names.index(active_joint)]
            # Evaluate quartic polynomial
            y    = self.eval_quartic(x=x,coef=coef,x0=x0,y0=y0)
            # Set passive joint position
            qrev[self.rev_joint_names.index(passive_joint)] = y

        # Return
        return qrev

    def get_subtree_body_names(
            self,
            root_body_name,
            include_root = False,
        ):
        """
        Get the names of all bodies in the subtree rooted at the specified body or bodies.

        Parameters:
            root_body_name (str or list): Name of the root body, or names of root bodies.
            include_root (bool): Whether to include each root body in the returned list.
        """
        if isinstance(root_body_name,str):
            root_body_names = [root_body_name]
        else:
            root_body_names = list(root_body_name)

        body_names = []
        for root_body_name in root_body_names:
            root_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,root_body_name)
            if root_id==-1:
                raise ValueError(f"Body '{root_body_name}' not found.")

            if include_root:
                body_names.append(root_body_name)

            stack = [root_id]
            while stack:
                pid = stack.pop()
                for cid in self.body_children_list[pid]:
                    cname = self.body_names[cid]
                    if cname is None:
                        continue
                    body_names.append(cname)
                    stack.append(cid)

        body_names = list(dict.fromkeys(body_names))
        return body_names
    
    def clear_obj_pcd_cache(self,obj_body_names=None):
        """
        Clear cached object-local point clouds.

        Parameters:
            obj_body_names (list | None):
                If None, clear all.
                Otherwise clear only specified body names.
        """
        cache_attr_list = [attr for attr in dir(self) if attr.startswith("obj_pcd_local_cache")]
        if len(cache_attr_list) == 0:
            self.obj_pcd_local_cache = {}
            return

        for cache_attr in cache_attr_list:
            cache = getattr(self,cache_attr)
            if obj_body_names is None:
                setattr(self,cache_attr,{})
            else:
                for obj_body_name in obj_body_names:
                    cache.pop(obj_body_name,None)

    @staticmethod
    def compute_view_params(
            camera_pos,
            target_pos,
            up_vector = np.array([0,0,1]), # Z-up convention
        ):
        """
        Compute view parameters (azimuth, distance, elevation, and lookat point) for a camera in 3D space.

        Parameters:
            camera_pos (np.ndarray): 3D position of the camera.
            target_pos (np.ndarray): 3D position of the target.
            up_vector (np.ndarray): 3D up vector (default is [0, 0, 1]). # Z-up convention

        Returns:
            tuple: (azimuth [deg], distance, elevation [deg], lookat point)
        """
        # Compute camera-to-target vector and distance
        cam_to_target = target_pos - camera_pos
        distance = np.linalg.norm(cam_to_target)

        # Compute azimuth and elevation
        azimuth   = np.arctan2(cam_to_target[1], cam_to_target[0])
        azimuth   = np.rad2deg(azimuth) # [deg]
        elevation = np.arcsin(cam_to_target[2] / distance)
        elevation = np.rad2deg(elevation) # [deg]

        # Compute lookat point
        lookat = target_pos

        # Compute camera orientation matrix
        zaxis = cam_to_target / distance
        xaxis = np.cross(up_vector, zaxis)
        yaxis = np.cross(zaxis, xaxis)
        cam_orient = np.array([xaxis, yaxis, zaxis]) # 3 x 3

        # Return computed values
        return azimuth, distance, elevation, lookat    
    
    def get_T_viewer(self):
        """
        Compute the viewer camera's 4x4 transform.
        This function assumes the camera looks along the -X axis.

        Returns:
            np.array: 4x4 homogeneous transform representing the camera pose.
        """
        cam_lookat    = self.viewer.cam.lookat
        cam_elevation = self.viewer.cam.elevation
        cam_azimuth   = self.viewer.cam.azimuth
        cam_distance  = self.viewer.cam.distance

        p_lookat = cam_lookat
        R_lookat = rpy2r(np.deg2rad([0,-cam_elevation,cam_azimuth])) # Z-up convention
        T_lookat = pr2t(p_lookat,R_lookat)
        T_viewer = T_lookat @ pr2t(np.array([-cam_distance,0,0]),np.eye(3)) # camera looks along -X axis
        return T_viewer
    
    def get_pcd_from_depth_img(self,depth_img,fovy=45):
        """
        Convert a depth image into a 3D point cloud in world coordinates.

        Parameters:
            depth_img (np.array): Depth image (H x W).
            fovy (float): Field of view angle in degrees for intrinsic approximation.

        Returns:
            tuple:
                - pcd (np.array): Nx3 array of 3D points.
                - xyz_img (np.array): HxWx3 array of camera-frame 3D points.
                - xyz_img_world (np.array): HxWx3 array of world-frame 3D points.
        """
        # Get camera pose
        T_viewer = self.get_T_viewer() # [4 x 4]

        # Camera intrinsic
        img_height = depth_img.shape[0]
        img_width  = depth_img.shape[1]
        focal_scaling = 0.5*img_height/np.tan(fovy*np.pi/360)
        cam_matrix = np.array(
            ((focal_scaling,0,img_width/2),
             (0,focal_scaling,img_height/2),
             (0,0,1))
             ) # [3 x 3]

        # Estimate 3D point from depth image
        xyz_img          = meters2xyz(depth_img,cam_matrix) # [H x W x 3]
        xyz_transpose    = np.transpose(xyz_img,(2,0,1)).reshape(3,-1) # [3 x N]
        xyzone_transpose = np.vstack((xyz_transpose,np.ones((1,xyz_transpose.shape[1])))) # [4 x N]

        # To world coordinate
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=RuntimeWarning)
            xyzone_world_transpose = T_viewer @ xyzone_transpose # [4 x 4] @ [4 x N]  => [4 x N]
            xyz_world_transpose    = xyzone_world_transpose[:3,:] # [3 x N]
            xyz_world              = np.transpose(xyz_world_transpose,(1,0)) # [N x 3]
            xyz_img_world          = xyz_world.reshape(depth_img.shape[0],depth_img.shape[1],3) # [H x W x 3]

        # Return
        return xyz_world,xyz_img,xyz_img_world    
    
    def grab_rgbd_img(self):
        """
        Capture the current RGB and depth images from the viewer.

        Returns:
            tuple:
                - rgb_img (np.array): The captured RGB image (H x W x 3, uint8).
                - depth_img (np.array): The captured depth image (H x W, float32).
        """
        rgb_img   = np.zeros((self.viewer.viewport.height,self.viewer.viewport.width,3),dtype=np.uint8)
        depth_img = np.zeros((self.viewer.viewport.height,self.viewer.viewport.width,1), dtype=np.float32)
        mujoco.mjr_readPixels(rgb_img,depth_img,self.viewer.viewport,self.viewer.ctx)
        rgb_img,depth_img = np.flipud(rgb_img),np.flipud(depth_img) # flip up-down

        # Rescale depth image
        extent = self.model.stat.extent
        near   = self.model.vis.map.znear * extent
        far    = self.model.vis.map.zfar * extent
        scaled_depth_img = near / (1.0 - depth_img * (1.0 - near / far)) # in meters
        depth_img = scaled_depth_img.squeeze()
        return rgb_img,depth_img    
    
    def get_egocentric_rgbd_pcd(
            self,
            p_ego            = None,
            p_trgt           = None,
            rsz_rate_for_pcd = None,
            rsz_rate_for_img = None,
            fovy             = None,
            restore_view     = True,
        ):
        """
        Capture egocentric RGB/Depth images and generate a point cloud.

        Parameters:
            p_ego (np.array): Camera position.
            p_trgt (np.array): Look-at position (target).
            rsz_rate_for_pcd (float, optional): Scale factor for depth image before generating PCD.
            rsz_rate_for_img (float, optional): Scale factor for final output images.
            fovy (float, optional): Field of view. If None, uses the first camera's fov or 45 if not defined.
            restore_view (bool): Whether to restore the original camera view afterward.

        Returns:
            tuple:
                - rgb_img (np.array): Captured RGB image.
                - depth_img (np.array): Captured depth image.
                - pcd (np.array): Nx3 point cloud.
                - xyz_img (np.array): Depth image in camera coordinates (HxWx3).
                - xyz_img_world (np.array): Depth image in world coordinates (HxWx3).
        """
        if restore_view:
            # Backup camera information
            viewer_azimuth,viewer_distance,viewer_elevation,viewer_lookat = self.viewer.get_cam_info()

        if (p_ego is not None) and (p_trgt is not None):
            cam_azimuth,cam_distance,cam_elevation,cam_lookat = self.compute_view_params(
                camera_pos = p_ego,
                target_pos = p_trgt,
                up_vector  = np.array([0,0,1]),
            )
            self.viewer.set_cam_info(
                azimuth   = cam_azimuth,
                distance  = cam_distance,
                elevation = cam_elevation,
                lookat    = cam_lookat,
            )

        # Update viewer and render to ensure the latest view is captured
        mujoco.mj_forward(self.model,self.data) 
        mujoco.mjv_updateScene(
            self.model,self.data,self.viewer.vopt,self.viewer.pert,self.viewer.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,self.viewer.scn)
        mujoco.mjr_render(self.viewer.viewport,self.viewer.scn,self.viewer.ctx)            
        
        # Grab RGB and depth image
        rgb_img,depth_img = self.grab_rgbd_img() # get rgb and depth images

        # Resize depth image for reducing point clouds
        if rsz_rate_for_pcd is not None:
            h_rsz         = int(depth_img.shape[0]*rsz_rate_for_pcd)
            w_rsz         = int(depth_img.shape[1]*rsz_rate_for_pcd)
            depth_img_rsz = cv2.resize(depth_img,(w_rsz,h_rsz),interpolation=cv2.INTER_NEAREST)
        else:
            depth_img_rsz = depth_img

        # Get PCD
        if fovy is None:
            if len(self.model.cam_fovy)==0: fovy = 45.0 # if cam is not defined, use 45 deg (default value)
            else: fovy = self.model.cam_fovy[0] # otherwise use the fovy of the first camera
        pcd,xyz_img,xyz_img_world = self.get_pcd_from_depth_img(depth_img_rsz,fovy=fovy) # [N x 3]

        # Resize rgb_image and depth_img (optional)
        if rsz_rate_for_img is not None:
            h = int(rgb_img.shape[0]*rsz_rate_for_img)
            w = int(rgb_img.shape[1]*rsz_rate_for_img)
            rgb_img   = cv2.resize(rgb_img,(w,h),interpolation=cv2.INTER_NEAREST)
            depth_img = cv2.resize(depth_img,(w,h),interpolation=cv2.INTER_NEAREST)

        # Restore view
        if restore_view:
            # Restore camera information
            self.viewer.set_cam_info(
                azimuth   = viewer_azimuth,
                distance  = viewer_distance,
                elevation = viewer_elevation,
                lookat    = viewer_lookat,
            )
            mujoco.mj_forward(self.model,self.data) 
            mujoco.mjv_updateScene(
                self.model,self.data,self.viewer.vopt,self.viewer.pert,self.viewer.cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,self.viewer.scn)
            mujoco.mjr_render(self.viewer.viewport,self.viewer.scn,self.viewer.ctx)            

        return rgb_img,depth_img,pcd,xyz_img,xyz_img_world
