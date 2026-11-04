#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import mujoco
import mujoco.viewer
import cv2
import numpy as np

import robot
from src.cinta_transportadora import CintaTransportadora
from src.camara_inspeccion import CamaraInspeccion
from src.horno import Horno
from src.objeto_industrial import ObjetoIndustrial


# -----------------------------------------------------------------------------
# CONSTANTES GLOBALES DE LA PLANTA
# -----------------------------------------------------------------------------
CAM_W = 360
CAM_H = 260
CAM_FPS = 20.0

BELT_SPEED = 0.55
VEL_KP = 7.0
CENTER_KP = 12.0
MAX_FORCE = 12.0

BELT_HALF_LENGTH = 1.45
BELT_HALF_WIDTH = 0.28
BELT_Z = 0.12
BELT_TOP_Z = 0.18
LANE = 1.45
TURN_START = 1.08
CORNER_SWITCH = 1.27
RAIL_END_GAP = 0.06
RAIL_HALF_LENGTH = LANE - BELT_HALF_WIDTH - RAIL_END_GAP
RAIL_THICKNESS = 0.035
RAIL_HALF_HEIGHT = 0.10
RAIL_CENTER_Z = 0.22

CAM_HEIGHT = 0.92
CAM_FOVY = 32.0
CAM_LENS_SIDE_OFFSET = 0.10
CAM_POLE_SIDE_OFFSET = BELT_HALF_WIDTH + 0.45

CAM_OVEN_HEIGHT = 0.78
CAM_OVEN_SIDE_OFFSET = 0.46
CAM_OVEN_FOVY = 34.0

CAM_LEFT_INSPECTION_Y = 0.82
CAM_LEFT_HEAD_CLEARANCE = 0.20
CAM_LEFT_POLE_CLEARANCE = 0.52

LOCK_OBJECT_ROTATION = True
CYLINDER_OBJECT_IDS = {1, 3}
LOCK_CYLINDERS_TO_CENTERLINE = True

BRANCH_SPEED = 0.48
BRANCH_Y = 0.0
BRANCH_HALF_LENGTH = 1.15
BRANCH_HALF_WIDTH = 0.32
BRANCH_CENTER_X = -(LANE + BRANCH_HALF_LENGTH - 0.05)
BRANCH_OPEN_HALF = 0.38
BRANCH_RAIL_HALF = (RAIL_HALF_LENGTH - BRANCH_OPEN_HALF) / 2.0
BRANCH_RAIL_CENTER_Y = (RAIL_HALF_LENGTH + BRANCH_OPEN_HALF) / 2.0
BRANCH_RAIL_START_X = -2.00
BRANCH_RAIL_END_X = -3.48
BRANCH_RAIL_CENTER_X = 0.5 * (BRANCH_RAIL_START_X + BRANCH_RAIL_END_X)
BRANCH_RAIL_HALF_LENGTH = 0.5 * abs(BRANCH_RAIL_END_X - BRANCH_RAIL_START_X)

PICK_RAIL_OPEN_Y_MIN = -0.10
PICK_RAIL_OPEN_Y_MAX = 0.72
LEFT_RAIL_UPPER_CENTER_Y = 0.5 * (RAIL_HALF_LENGTH + PICK_RAIL_OPEN_Y_MAX)
LEFT_RAIL_UPPER_HALF = 0.5 * (RAIL_HALF_LENGTH - PICK_RAIL_OPEN_Y_MAX)
LEFT_RAIL_LOWER_CENTER_Y = 0.5 * (-RAIL_HALF_LENGTH + PICK_RAIL_OPEN_Y_MIN)
LEFT_RAIL_LOWER_HALF = 0.5 * (PICK_RAIL_OPEN_Y_MIN + RAIL_HALF_LENGTH)

ISLAND_X = -1.02
ISLAND_Y = 0.20
ISLAND_Z = 0.16
ISLAND_HALF_X = 0.18
ISLAND_HALF_Y = 0.38

OVEN_CENTER_X = -4.05
OVEN_ENTRY_X = -3.58

OBJECT_HALF = 0.085
OBJ1_RADIUS = 0.075
OBJ1_HALF_HEIGHT = 0.045
OBJ2_HALF_Z = 0.070
OBJ3_RADIUS = 0.085
OBJ3_HALF_HEIGHT = 0.0375
OBJ4_HALF_Z = 0.065
OBJECT_CLEARANCE_Z = 0.01
OBJECT_REST_Z = [
    BELT_TOP_Z + OBJECT_HALF + OBJECT_CLEARANCE_Z,
    BELT_TOP_Z + OBJ1_HALF_HEIGHT + OBJECT_CLEARANCE_Z,
    BELT_TOP_Z + OBJ2_HALF_Z + OBJECT_CLEARANCE_Z,
    BELT_TOP_Z + OBJ3_HALF_HEIGHT + OBJECT_CLEARANCE_Z,
    BELT_TOP_Z + OBJ4_HALF_Z + OBJECT_CLEARANCE_Z,
]

def camera_xyaxes(cam_pos, target):
    cam = np.asarray(cam_pos, dtype=float)
    tgt = np.asarray(target, dtype=float)
    z_axis = cam - tgt
    z_axis /= np.linalg.norm(z_axis)
    up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    vals = np.concatenate([x_axis, y_axis])
    return " ".join(f"{v:.6f}" for v in vals)

def build_mjcf():
    cam_bottom = (0.0, -(LANE + CAM_LENS_SIDE_OFFSET), CAM_HEIGHT)
    cam_right  = (LANE + CAM_LENS_SIDE_OFFSET, 0.0, CAM_HEIGHT)
    cam_top    = (0.0, LANE + CAM_LENS_SIDE_OFFSET, CAM_HEIGHT)
    cam_left   = (-(LANE + BELT_HALF_WIDTH + CAM_LEFT_HEAD_CLEARANCE), CAM_LEFT_INSPECTION_Y, CAM_HEIGHT)
    cam_oven   = (BRANCH_CENTER_X, BRANCH_Y - CAM_OVEN_SIDE_OFFSET, CAM_OVEN_HEIGHT)

    target_bottom = (0.0, -LANE, BELT_TOP_Z + 0.015)
    target_right  = (LANE, 0.0, BELT_TOP_Z + 0.015)
    target_top    = (0.0, LANE, BELT_TOP_Z + 0.015)
    target_left   = (-LANE, CAM_LEFT_INSPECTION_Y, BELT_TOP_Z + 0.015)
    target_oven   = (BRANCH_CENTER_X, BRANCH_Y, BELT_TOP_Z + 0.015)

    pole_bottom_y = -(LANE + CAM_POLE_SIDE_OFFSET)
    pole_right_x  =  (LANE + CAM_POLE_SIDE_OFFSET)
    pole_top_y    =  (LANE + CAM_POLE_SIDE_OFFSET)
    pole_left_x   = -(LANE + BELT_HALF_WIDTH + CAM_LEFT_POLE_CLEARANCE)
    pole_left_y   = CAM_LEFT_INSPECTION_Y
    pole_oven_y   = BRANCH_Y - (BRANCH_HALF_WIDTH + 0.52)

    xy_bottom = camera_xyaxes(cam_bottom, target_bottom)
    xy_right  = camera_xyaxes(cam_right, target_right)
    xy_top    = camera_xyaxes(cam_top, target_top)
    xy_left   = camera_xyaxes(cam_left, target_left)
    xy_oven   = camera_xyaxes(cam_oven, target_oven)

    xml = f"""
<mujoco model="square_conveyor_factory">
    <compiler angle="degree" autolimits="true"/>
    <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>

    <visual>
        <global offwidth="1280" offheight="960"/>
        <quality shadowsize="2048"/>
        <headlight ambient="0.35 0.35 0.35" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
    </visual>

    <default>
        <geom contype="1" conaffinity="1" friction="1.2 0.02 0.002" condim="4"/>
        <joint damping="0.15"/>
    </default>

    <asset>
        <texture name="floor_tex" type="2d" builtin="checker"
                 rgb1="0.18 0.20 0.22" rgb2="0.23 0.25 0.27" width="512" height="512"/>
        <material name="floor_mat" texture="floor_tex" texrepeat="8 8" reflectance="0.10"/>
        <material name="belt_mat" rgba="0.08 0.10 0.12 1"/>
        <material name="belt_edge_mat" rgba="0.28 0.32 0.36 1"/>
        <material name="transfer_mat" rgba="0.18 0.20 0.23 1"/>
        <material name="camera_mat" rgba="0.07 0.07 0.08 1"/>
        <material name="pole_mat" rgba="0.30 0.33 0.36 1"/>
        <material name="roller_mat" rgba="0.13 0.15 0.17 1"/>
        <material name="island_mat" rgba="0.24 0.28 0.31 1"/>
        <material name="island_panel_mat" rgba="0.10 0.38 0.52 1"/>
        <material name="oven_mat" rgba="0.22 0.24 0.26 1"/>
        <material name="oven_dark_mat" rgba="0.06 0.06 0.07 1"/>
        <material name="oven_hot_mat" rgba="0.95 0.22 0.05 1" emission="0.8"/>
    </asset>

    <worldbody>
        <light pos="0 0 6" dir="0 0 -1" diffuse="0.85 0.85 0.85"/>
        <light pos="3 -3 4" dir="-0.5 0.5 -1" diffuse="0.45 0.45 0.45"/>
        <geom name="floor" type="plane" size="6 6 0.1" material="floor_mat"/>

        <!-- CINTA 1: INFERIOR, flujo +X -->
        <geom name="belt_bottom" type="box" pos="0 -{LANE} {BELT_Z}"
              size="{BELT_HALF_LENGTH} {BELT_HALF_WIDTH} 0.06" material="belt_mat"/>
        <geom type="box" pos="0 -{LANE-BELT_HALF_WIDTH-RAIL_THICKNESS} {RAIL_CENTER_Z}"
              size="{RAIL_HALF_LENGTH} {RAIL_THICKNESS} {RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
        <geom type="box" pos="0 -{LANE+BELT_HALF_WIDTH+RAIL_THICKNESS} {RAIL_CENTER_Z}"
              size="{RAIL_HALF_LENGTH} {RAIL_THICKNESS} {RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>

        <!-- CINTA 2: DERECHA, flujo +Y -->
        <geom name="belt_right" type="box" pos="{LANE} 0 {BELT_Z}"
              size="{BELT_HALF_WIDTH} {BELT_HALF_LENGTH} 0.06" material="belt_mat"/>
        <geom type="box" pos="{LANE-BELT_HALF_WIDTH-RAIL_THICKNESS} 0 {RAIL_CENTER_Z}"
              size="{RAIL_THICKNESS} {RAIL_HALF_LENGTH} {RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
        <geom type="box" pos="{LANE+BELT_HALF_WIDTH+RAIL_THICKNESS} 0 {RAIL_CENTER_Z}"
              size="{RAIL_THICKNESS} {RAIL_HALF_LENGTH} {RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>

        <!-- CINTA 3: SUPERIOR, flujo -X -->
        <geom name="belt_top" type="box" pos="0 {LANE} {BELT_Z}"
              size="{BELT_HALF_LENGTH} {BELT_HALF_WIDTH} 0.06" material="belt_mat"/>
        <geom type="box" pos="0 {LANE-BELT_HALF_WIDTH-RAIL_THICKNESS} {RAIL_CENTER_Z}"
              size="{RAIL_HALF_LENGTH} {RAIL_THICKNESS} {RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
        <geom type="box" pos="0 {LANE+BELT_HALF_WIDTH+RAIL_THICKNESS} {RAIL_CENTER_Z}"
              size="{RAIL_HALF_LENGTH} {RAIL_THICKNESS} {RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>

        <!-- CINTA 4: IZQUIERDA, flujo -Y -->
        <geom name="belt_left" type="box" pos="-{LANE} 0 {BELT_Z}"
              size="{BELT_HALF_WIDTH} {BELT_HALF_LENGTH} 0.06" material="belt_mat"/>

        <geom type="box" pos="-{LANE-BELT_HALF_WIDTH-RAIL_THICKNESS} {LEFT_RAIL_UPPER_CENTER_Y} {RAIL_CENTER_Z}"
              size="{RAIL_THICKNESS} {LEFT_RAIL_UPPER_HALF} {RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
        <geom type="box" pos="-{LANE-BELT_HALF_WIDTH-RAIL_THICKNESS} {LEFT_RAIL_LOWER_CENTER_Y} {RAIL_CENTER_Z}"
              size="{RAIL_THICKNESS} {LEFT_RAIL_LOWER_HALF} {RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>

        <geom type="box" pos="-{LANE+BELT_HALF_WIDTH+RAIL_THICKNESS} {BRANCH_RAIL_CENTER_Y} {RAIL_CENTER_Z}"
              size="{RAIL_THICKNESS} {BRANCH_RAIL_HALF} {RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
        <geom type="box" pos="-{LANE+BELT_HALF_WIDTH+RAIL_THICKNESS} -{BRANCH_RAIL_CENTER_Y} {RAIL_CENTER_Z}"
              size="{RAIL_THICKNESS} {BRANCH_RAIL_HALF} {RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>

        <!-- CINTA 5: RAMA LATERAL HACIA EL HORNO -->
        <geom name="belt_oven" type="box" pos="{BRANCH_CENTER_X} {BRANCH_Y} {BELT_Z}"
              size="{BRANCH_HALF_LENGTH} {BRANCH_HALF_WIDTH} 0.06" material="belt_mat"/>
        <geom name="branch_transfer" type="box" pos="-{LANE} {BRANCH_Y} {BELT_Z}"
              size="{BELT_HALF_WIDTH+0.06} {BRANCH_HALF_WIDTH} 0.06" material="transfer_mat"/>

        <geom type="box" pos="{BRANCH_RAIL_CENTER_X} {BRANCH_Y-BRANCH_HALF_WIDTH-RAIL_THICKNESS} {RAIL_CENTER_Z}"
              size="{BRANCH_RAIL_HALF_LENGTH} {RAIL_THICKNESS} {RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>
        <geom type="box" pos="{BRANCH_RAIL_CENTER_X} {BRANCH_Y+BRANCH_HALF_WIDTH+RAIL_THICKNESS} {RAIL_CENTER_Z}"
              size="{BRANCH_RAIL_HALF_LENGTH} {RAIL_THICKNESS} {RAIL_HALF_HEIGHT}" material="belt_edge_mat"/>

        <!-- ISLA ROBOTIZADA -->
        <body name="process_island" pos="{ISLAND_X} {ISLAND_Y} 0">
            <geom name="island_base" type="box" pos="0 0 {ISLAND_Z/2}"
                  size="{ISLAND_HALF_X} {ISLAND_HALF_Y} {ISLAND_Z/2}"
                  material="island_mat" contype="0" conaffinity="0"/>
            <geom name="island_panda_plate" type="cylinder" pos="0 0 {ISLAND_Z+0.025}"
                  size="0.16 0.025" material="belt_edge_mat" contype="0" conaffinity="0"/>
        </body>

        <!-- HORNO INDUSTRIAL -->
        <body name="oven" pos="{OVEN_CENTER_X} 0 0">
            <geom type="box" pos="0 0 0.12" size="0.55 0.62 0.10" material="oven_mat"/>
            <geom type="box" pos="-0.50 0 0.68" size="0.08 0.62 0.58" material="oven_mat"/>
            <geom type="box" pos="0 -0.56 0.68" size="0.55 0.06 0.58" material="oven_mat"/>
            <geom type="box" pos="0  0.56 0.68" size="0.55 0.06 0.58" material="oven_mat"/>
            <geom type="box" pos="0 0 1.23" size="0.55 0.62 0.07" material="oven_mat"/>
            <geom type="box" pos="0.50 -0.48 0.68" size="0.05 0.08 0.58" material="oven_dark_mat"/>
            <geom type="box" pos="0.50  0.48 0.68" size="0.05 0.08 0.58" material="oven_dark_mat"/>
            <geom type="box" pos="0.50 0 1.16" size="0.05 0.40 0.10" material="oven_dark_mat"/>
            <geom type="box" pos="-0.42 0 0.58" size="0.025 0.39 0.31" material="oven_hot_mat" contype="0" conaffinity="0"/>
            <light pos="-0.18 0 0.82" dir="1 0 -0.15" diffuse="1.0 0.25 0.05"/>
        </body>

        <!-- ESQUINAS DE TRANSFERENCIA -->
        <geom name="corner_br" type="box" pos=" {LANE} -{LANE} {BELT_Z}"
              size="{BELT_HALF_WIDTH} {BELT_HALF_WIDTH} 0.06" material="transfer_mat"/>
        <geom name="corner_tr" type="box" pos=" {LANE}  {LANE} {BELT_Z}"
              size="{BELT_HALF_WIDTH} {BELT_HALF_WIDTH} 0.06" material="transfer_mat"/>
        <geom name="corner_tl" type="box" pos="-{LANE}  {LANE} {BELT_Z}"
              size="{BELT_HALF_WIDTH} {BELT_HALF_WIDTH} 0.06" material="transfer_mat"/>
        <geom name="corner_bl" type="box" pos="-{LANE} -{LANE} {BELT_Z}"
              size="{BELT_HALF_WIDTH} {BELT_HALF_WIDTH} 0.06" material="transfer_mat"/>

        <!-- RODILLOS VISUALES -->
        <body name="roller_b1" pos="-1.15 -{LANE} 0.20">
            <joint name="jr_b1" type="hinge" axis="0 1 0"/>
            <geom type="cylinder" size="0.08 0.25" euler="90 0 0" material="roller_mat" contype="0" conaffinity="0"/>
        </body>
        <body name="roller_b2" pos="1.15 -{LANE} 0.20">
            <joint name="jr_b2" type="hinge" axis="0 1 0"/>
            <geom type="cylinder" size="0.08 0.25" euler="90 0 0" material="roller_mat" contype="0" conaffinity="0"/>
        </body>
        <body name="roller_r1" pos="{LANE} -1.15 0.20">
            <joint name="jr_r1" type="hinge" axis="1 0 0"/>
            <geom type="cylinder" size="0.08 0.25" euler="0 90 0" material="roller_mat" contype="0" conaffinity="0"/>
        </body>
        <body name="roller_r2" pos="{LANE} 1.15 0.20">
            <joint name="jr_r2" type="hinge" axis="1 0 0"/>
            <geom type="cylinder" size="0.08 0.25" euler="0 90 0" material="roller_mat" contype="0" conaffinity="0"/>
        </body>
        <body name="roller_t1" pos="1.15 {LANE} 0.20">
            <joint name="jr_t1" type="hinge" axis="0 1 0"/>
            <geom type="cylinder" size="0.08 0.25" euler="90 0 0" material="roller_mat" contype="0" conaffinity="0"/>
        </body>
        <body name="roller_t2" pos="-1.15 {LANE} 0.20">
            <joint name="jr_t2" type="hinge" axis="0 1 0"/>
            <geom type="cylinder" size="0.08 0.25" euler="90 0 0" material="roller_mat" contype="0" conaffinity="0"/>
        </body>
        <body name="roller_l1" pos="-{LANE} 1.15 0.20">
            <joint name="jr_l1" type="hinge" axis="1 0 0"/>
            <geom type="cylinder" size="0.08 0.25" euler="0 90 0" material="roller_mat" contype="0" conaffinity="0"/>
        </body>
        <body name="roller_l2" pos="-{LANE} -1.15 0.20">
            <joint name="jr_l2" type="hinge" axis="1 0 0"/>
            <geom type="cylinder" size="0.08 0.25" euler="0 90 0" material="roller_mat" contype="0" conaffinity="0"/>
        </body>
        <body name="roller_o1" pos="-2.18 {BRANCH_Y} 0.20">
            <joint name="jr_o1" type="hinge" axis="0 1 0"/>
            <geom type="cylinder" size="0.08 0.25" euler="90 0 0" material="roller_mat" contype="0" conaffinity="0"/>
        </body>
        <body name="roller_o2" pos="-3.25 {BRANCH_Y} 0.20">
            <joint name="jr_o2" type="hinge" axis="0 1 0"/>
            <geom type="cylinder" size="0.08 0.25" euler="90 0 0" material="roller_mat" contype="0" conaffinity="0"/>
        </body>

        <!-- CÁMARAS DE INSPECCIÓN -->
        <body name="cam_station_bottom" pos="0 0 0">
            <geom type="cylinder" pos="0 {pole_bottom_y} {CAM_HEIGHT/2}"
                  size="0.025 {CAM_HEIGHT/2}" material="pole_mat" contype="0" conaffinity="0"/>
            <geom type="box" pos="0 {(pole_bottom_y + cam_bottom[1])/2} {CAM_HEIGHT}"
                  size="0.025 {abs(cam_bottom[1]-pole_bottom_y)/2} 0.022" material="pole_mat" contype="0" conaffinity="0"/>
            <geom type="box" pos="{cam_bottom[0]} {cam_bottom[1]} {cam_bottom[2]}"
                  size="0.075 0.055 0.055" material="camera_mat" contype="0" conaffinity="0"/>
            <camera name="cam_bottom" pos="{cam_bottom[0]} {cam_bottom[1]} {cam_bottom[2]}"
                    xyaxes="{xy_bottom}" fovy="{CAM_FOVY}"/>
        </body>

        <body name="cam_station_right" pos="0 0 0">
            <geom type="cylinder" pos="{pole_right_x} 0 {CAM_HEIGHT/2}"
                  size="0.025 {CAM_HEIGHT/2}" material="pole_mat" contype="0" conaffinity="0"/>
            <geom type="box" pos="{(pole_right_x + cam_right[0])/2} 0 {CAM_HEIGHT}"
                  size="{abs(cam_right[0]-pole_right_x)/2} 0.025 0.022" material="pole_mat" contype="0" conaffinity="0"/>
            <geom type="box" pos="{cam_right[0]} {cam_right[1]} {cam_right[2]}"
                  size="0.055 0.075 0.055" material="camera_mat" contype="0" conaffinity="0"/>
            <camera name="cam_right" pos="{cam_right[0]} {cam_right[1]} {cam_right[2]}"
                    xyaxes="{xy_right}" fovy="{CAM_FOVY}"/>
        </body>

        <body name="cam_station_top" pos="0 0 0">
            <geom type="cylinder" pos="0 {pole_top_y} {CAM_HEIGHT/2}"
                  size="0.025 {CAM_HEIGHT/2}" material="pole_mat" contype="0" conaffinity="0"/>
            <geom type="box" pos="0 {(pole_top_y + cam_top[1])/2} {CAM_HEIGHT}"
                  size="0.025 {abs(cam_top[1]-pole_top_y)/2} 0.022" material="pole_mat" contype="0" conaffinity="0"/>
            <geom type="box" pos="{cam_top[0]} {cam_top[1]} {cam_top[2]}"
                  size="0.075 0.055 0.055" material="camera_mat" contype="0" conaffinity="0"/>
            <camera name="cam_top" pos="{cam_top[0]} {cam_top[1]} {cam_top[2]}"
                    xyaxes="{xy_top}" fovy="{CAM_FOVY}"/>
        </body>

        <body name="cam_station_left" pos="0 0 0">
            <geom type="cylinder" pos="{pole_left_x} {pole_left_y} {CAM_HEIGHT/2}"
                  size="0.025 {CAM_HEIGHT/2}" material="pole_mat" contype="0" conaffinity="0"/>
            <geom type="box" pos="{(pole_left_x + cam_left[0])/2} {pole_left_y} {CAM_HEIGHT}"
                  size="{abs(cam_left[0]-pole_left_x)/2} 0.025 0.022" material="pole_mat" contype="0" conaffinity="0"/>
            <geom type="box" pos="{cam_left[0]} {cam_left[1]} {cam_left[2]}"
                  size="0.055 0.075 0.055" material="camera_mat" contype="0" conaffinity="0"/>
            <camera name="cam_left" pos="{cam_left[0]} {cam_left[1]} {cam_left[2]}"
                    xyaxes="{xy_left}" fovy="{CAM_FOVY}"/>
        </body>

        <body name="cam_station_oven" pos="0 0 0">
            <geom type="cylinder" pos="{BRANCH_CENTER_X} {pole_oven_y} {CAM_OVEN_HEIGHT/2}"
                  size="0.025 {CAM_OVEN_HEIGHT/2}" material="pole_mat" contype="0" conaffinity="0"/>
            <geom type="box" pos="{BRANCH_CENTER_X} {(pole_oven_y + cam_oven[1])/2} {CAM_OVEN_HEIGHT}"
                  size="0.025 {abs(cam_oven[1]-pole_oven_y)/2} 0.022" material="pole_mat" contype="0" conaffinity="0"/>
            <geom type="box" pos="{cam_oven[0]} {cam_oven[1]} {cam_oven[2]}"
                  size="0.075 0.055 0.055" material="camera_mat" contype="0" conaffinity="0"/>
            <camera name="cam_oven" pos="{cam_oven[0]} {cam_oven[1]} {cam_oven[2]}"
                    xyaxes="{xy_oven}" fovy="{CAM_OVEN_FOVY}"/>
        </body>

        <camera name="overview" pos="4.6 -4.6 5.2" xyaxes="0.707 0.707 0 -0.49 0.49 0.72" fovy="50"/>

        <!-- OBJETOS -->
        <body name="obj0" pos="-0.95 -{LANE} {OBJECT_REST_Z[0]}">
            <freejoint name="obj0_free"/>
            <geom name="obj0_geom" type="box" size="{OBJECT_HALF} {OBJECT_HALF} {OBJECT_HALF}"
                  mass="0.18" rgba="0.90 0.16 0.12 1"/>
        </body>
        <body name="obj1" pos="{LANE} -0.70 {OBJECT_REST_Z[1]}">
            <freejoint name="obj1_free"/>
            <geom name="obj1_geom" type="cylinder" size="{OBJ1_RADIUS} {OBJ1_HALF_HEIGHT}"
                  mass="0.08" rgba="0.12 0.45 0.95 1"/>
        </body>
        <body name="obj2" pos="0.55 {LANE} {OBJECT_REST_Z[2]}">
            <freejoint name="obj2_free"/>
            <geom name="obj2_geom" type="box" size="0.10 0.07 0.07"
                  mass="0.15" rgba="0.95 0.76 0.10 1"/>
        </body>
        <body name="obj3" pos="-{LANE} 0.45 {OBJECT_REST_Z[3]}">
            <freejoint name="obj3_free"/>
            <geom name="obj3_geom" type="cylinder" size="{OBJ3_RADIUS} {OBJ3_HALF_HEIGHT}"
                  mass="0.085" rgba="0.18 0.75 0.30 1"/>
        </body>
        <body name="obj4" pos="0.15 -{LANE} {OBJECT_REST_Z[4]}">
            <freejoint name="obj4_free"/>
            <geom name="obj4_geom" type="box" size="0.07 0.11 0.065"
                  mass="0.14" rgba="0.72 0.28 0.90 1"/>
        </body>
    </worldbody>

    <actuator>
        <velocity name="belt_act_b1" joint="jr_b1" kv="3" ctrlrange="-9 9"/>
        <velocity name="belt_act_b2" joint="jr_b2" kv="3" ctrlrange="-9 9"/>
        <velocity name="belt_act_r1" joint="jr_r1" kv="3" ctrlrange="-9 9"/>
        <velocity name="belt_act_r2" joint="jr_r2" kv="3" ctrlrange="-9 9"/>
        <velocity name="belt_act_t1" joint="jr_t1" kv="3" ctrlrange="-9 9"/>
        <velocity name="belt_act_t2" joint="jr_t2" kv="3" ctrlrange="-9 9"/>
        <velocity name="belt_act_l1" joint="jr_l1" kv="3" ctrlrange="-9 9"/>
        <velocity name="belt_act_l2" joint="jr_l2" kv="3" ctrlrange="-9 9"/>
        <velocity name="belt_act_o1" joint="jr_o1" kv="3" ctrlrange="-9 9"/>
        <velocity name="belt_act_o2" joint="jr_o2" kv="3" ctrlrange="-9 9"/>
    </actuator>
</mujoco>
"""
    return xml


class PlantaIndustrial:
    def __init__(self):
        self.VEL_KP = 7.0
        self.CENTER_KP = 12.0
        self.MAX_FORCE = 12.0
        self.LANE = 1.45
        self.TURN_START = 1.08
        self.CORNER_SWITCH = 1.27
        self.BELT_SPEED = 0.55
        self.BRANCH_SPEED = 0.48
        self.BRANCH_Y = 0.0
        self.OVEN_ENTRY_X = -3.58

        self.ROBOT_CONFIG = robot.make_config(
            lane=self.LANE,
            branch_y=self.BRANCH_Y,
            branch_speed=self.BRANCH_SPEED,
        )

        self.cintas = [
            CintaTransportadora(1, "Inferior", self.BELT_SPEED),
            CintaTransportadora(2, "Derecha", self.BELT_SPEED),
            CintaTransportadora(3, "Superior", self.BELT_SPEED),
            CintaTransportadora(4, "Izquierda", self.BELT_SPEED),
            CintaTransportadora(5, "Horno", self.BRANCH_SPEED)
        ]
        self.objetos = [
            ObjetoIndustrial(0, "caja", "obj0"),
            ObjetoIndustrial(1, "cilindro", "obj1"),
            ObjetoIndustrial(2, "caja", "obj2"),
            ObjetoIndustrial(3, "cilindro", "obj3"),
            ObjetoIndustrial(4, "caja", "obj4")
        ]
        
        self.horno = Horno(geom_id=None, temperatura_inicial=20.0)
        
        self.camaras = [
            CamaraInspeccion("cam_bottom", "CINTA 1 · INFERIOR · +X", self.cintas[0]),
            CamaraInspeccion("cam_right", "CINTA 2 · DERECHA · +Y", self.cintas[1]),
            CamaraInspeccion("cam_top", "CINTA 3 · SUPERIOR · -X", self.cintas[2]),
            CamaraInspeccion("cam_left", "CINTA 4 · IZQUIERDA · -Y", self.cintas[3]),
            CamaraInspeccion("cam_oven", "CINTA 5 · HORNO · -X", self.cintas[4]),
            CamaraInspeccion("overview", "VISTA GENERAL", None)
        ]
        
        self.model = None
        self.data = None
        self.renderer = None
        self.paused = False
        
        self.body_ids = []
        self.dof_addrs = []
        self.segments = [3, 1, 2, 3, 0]
        self.oven_counts = [0, 0, 0, 0, 0]
        self.robot_ids = None
        self.robot_state = robot.new_robot_state()

    def inicializar(self):
        xml_base = self.construir_xml_planta()
        self.model = robot.build_model_with_franka(xml_base, self.ROBOT_CONFIG)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model)
        
        # Enlazar el material incandescente del horno para cambiar su color dinámicamente
        for m in range(self.model.nmat):
            mat_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, m)
            if mat_name == "oven_hot_mat":
                self.horno.mat_id = m
                break

        self.body_ids, self.dof_addrs = self.get_ids(self.model)
        self.robot_ids = robot.get_robot_ids(self.model)
        
        self.reset_objects()
        robot.initialize_robot(self.model, self.data, self.robot_ids, self.robot_state)
        self.actualizar_actuadores_cintas()

    def get_ids(self, model):
        body_ids = []
        dof_addrs = []
        for i in range(5):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obj{i}")
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
            body_ids.append(bid)
            dof_addrs.append(model.jnt_dofadr[jid])
        return body_ids, dof_addrs

    def reset_objects(self):
        starts = [
            (-self.LANE, 0.95, OBJECT_REST_Z[0], 3),
            (self.LANE, -0.72, OBJECT_REST_Z[1], 1),
            (0.58, self.LANE, OBJECT_REST_Z[2], 2),
            (-0.62, -self.LANE, OBJECT_REST_Z[3], 0),
            (-0.45, self.LANE, OBJECT_REST_Z[4], 2),
        ]
        for i, (x, y, z, seg) in enumerate(starts):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
            qadr = self.model.jnt_qposadr[jid]
            dadr = self.model.jnt_dofadr[jid]
            self.data.qpos[qadr:qadr+3] = [x, y, z]
            self.data.qpos[qadr+3:qadr+7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qvel[dadr:dadr+6] = 0.0
            self.segments[i] = seg
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def lock_object_rotations(self):
        if not LOCK_OBJECT_ROTATION:
            return
        for i in range(5):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
            qadr = self.model.jnt_qposadr[jid]
            dadr = self.model.jnt_dofadr[jid]
            self.data.qpos[qadr+3:qadr+7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qvel[dadr+3:dadr+6] = 0.0

    def lock_cylinders_to_belt_center(self):
        if not LOCK_CYLINDERS_TO_CENTERLINE:
            return
        for i in CYLINDER_OBJECT_IDS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
            qadr = self.model.jnt_qposadr[jid]
            dadr = self.model.jnt_dofadr[jid]
            x = float(self.data.qpos[qadr])
            y = float(self.data.qpos[qadr + 1])
            seg = self.segments[i]

            if seg == 0 and x < self.TURN_START:
                self.data.qpos[qadr + 1] = -self.LANE
                self.data.qvel[dadr + 1] = 0.0
            elif seg == 1 and y < self.TURN_START:
                self.data.qpos[qadr] = self.LANE
                self.data.qvel[dadr] = 0.0
            elif seg == 2 and x > -self.TURN_START:
                self.data.qpos[qadr + 1] = self.LANE
                self.data.qvel[dadr + 1] = 0.0
            elif seg == 3 and y > -self.TURN_START:
                self.data.qpos[qadr] = -self.LANE
                self.data.qvel[dadr] = 0.0
            elif seg == 4:
                self.data.qpos[qadr + 1] = self.BRANCH_Y
                self.data.qvel[dadr + 1] = 0.0

    def actualizar_actuadores_cintas(self):
        f1 = self.cintas[0].esta_activa()
        f2 = self.cintas[1].esta_activa()
        f3 = self.cintas[2].esta_activa()
        f4 = self.cintas[3].esta_activa()
        f5 = self.cintas[4].esta_activa()
        
        roller_commands = {
            "belt_act_b1":  5.5 if f1 else 0.0,
            "belt_act_b2":  5.5 if f1 else 0.0,
            "belt_act_r1": -5.5 if f2 else 0.0,
            "belt_act_r2": -5.5 if f2 else 0.0,
            "belt_act_t1": -5.5 if f3 else 0.0,
            "belt_act_t2": -5.5 if f3 else 0.0,
            "belt_act_l1":  5.5 if f4 else 0.0,
            "belt_act_l2":  5.5 if f4 else 0.0,
            "belt_act_o1": -5.5 if f5 else 0.0,
            "belt_act_o2": -5.5 if f5 else 0.0,
        }
        for name, value in roller_commands.items():
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                self.data.ctrl[aid] = value

    def update_segment(self, pos, seg):
        x, y = float(pos[0]), float(pos[1])
        if seg == 0 and x >= self.CORNER_SWITCH: return 1
        if seg == 1 and y >= self.CORNER_SWITCH: return 2
        if seg == 2 and x <= -self.CORNER_SWITCH: return 3
        if seg == 3 and y <= -self.CORNER_SWITCH: return 0
        return seg

    def transport_force(self, pos, vel, seg):
        x, y = float(pos[0]), float(pos[1])
        vx, vy = float(vel[0]), float(vel[1])
        fx, fy = 0.0, 0.0

        cinta_activa = self.cintas[seg].esta_activa() if seg < len(self.cintas) else False
        if not cinta_activa:
            fx = self.VEL_KP * (0.0 - vx)
            fy = self.VEL_KP * (0.0 - vy)
            return float(np.clip(fx, -self.MAX_FORCE, self.MAX_FORCE)), float(np.clip(fy, -self.MAX_FORCE, self.MAX_FORCE))

        if seg == 0:
            alpha = np.clip((x - self.TURN_START) / max(self.CORNER_SWITCH - self.TURN_START, 1e-6), 0.0, 1.0)
            desired_vx = self.BELT_SPEED * (1.0 - 0.45 * alpha)
            desired_vy = self.BELT_SPEED * (0.85 * alpha)
            fx = self.VEL_KP * (desired_vx - vx)
            fy = self.VEL_KP * (desired_vy - vy) + (1.0 - alpha) * self.CENTER_KP * (-self.LANE - y)
        elif seg == 1:
            alpha = np.clip((y - self.TURN_START) / max(self.CORNER_SWITCH - self.TURN_START, 1e-6), 0.0, 1.0)
            desired_vx = -self.BELT_SPEED * (0.85 * alpha)
            desired_vy = self.BELT_SPEED * (1.0 - 0.45 * alpha)
            fx = self.VEL_KP * (desired_vx - vx) + (1.0 - alpha) * self.CENTER_KP * (self.LANE - x)
            fy = self.VEL_KP * (desired_vy - vy)
        elif seg == 2:
            alpha = np.clip((-x - self.TURN_START) / max(self.CORNER_SWITCH - self.TURN_START, 1e-6), 0.0, 1.0)
            desired_vx = -self.BELT_SPEED * (1.0 - 0.45 * alpha)
            desired_vy = -self.BELT_SPEED * (0.85 * alpha)
            fx = self.VEL_KP * (desired_vx - vx)
            fy = self.VEL_KP * (desired_vy - vy) + (1.0 - alpha) * self.CENTER_KP * (self.LANE - y)
        elif seg == 3:
            alpha = np.clip((-y - self.TURN_START) / max(self.CORNER_SWITCH - self.TURN_START, 1e-6), 0.0, 1.0)
            desired_vx = self.BELT_SPEED * (0.85 * alpha)
            desired_vy = -self.BELT_SPEED * (1.0 - 0.45 * alpha)
            fx = self.VEL_KP * (desired_vx - vx) + (1.0 - alpha) * self.CENTER_KP * (-self.LANE - x)
            fy = self.VEL_KP * (desired_vy - vy)
        elif seg == 4:
            desired_vx = -self.BRANCH_SPEED
            desired_vy = 0.0
            fx = self.VEL_KP * (desired_vx - vx)
            fy = self.VEL_KP * (desired_vy - vy) + self.CENTER_KP * (self.BRANCH_Y - y)

        return float(np.clip(fx, -self.MAX_FORCE, self.MAX_FORCE)), float(np.clip(fy, -self.MAX_FORCE, self.MAX_FORCE))

    def actualizar(self):
        if not self.paused and self.model and self.data:
            self.data.xfrc_applied[:] = 0.0

            robot.update_robot_state_machine(
                self.model, self.data, self.body_ids, self.robot_ids, 
                self.robot_state, self.segments, OBJECT_REST_Z, self.ROBOT_CONFIG
            )

            for i, bid in enumerate(self.body_ids):
                pos = self.data.xpos[bid]
                dadr = self.dof_addrs[i]
                vel = self.data.qvel[dadr:dadr+3]

                if i == self.robot_state["grasped_obj"]:
                    continue

                if i == self.robot_state["target_obj"] and self.robot_state["state"] != "idle":
                    fx, fy = robot.robot_hold_force(pos, vel, self.ROBOT_CONFIG, self.MAX_FORCE)
                else:
                    self.segments[i] = self.update_segment(pos, self.segments[i])
                    fx, fy = self.transport_force(pos, vel, self.segments[i])

                self.data.xfrc_applied[bid, 0] = fx
                self.data.xfrc_applied[bid, 1] = fy

                if self.segments[i] == 4 and pos[0] <= self.OVEN_ENTRY_X:
                    self.oven_counts[i] += 1
                    self.horno.registrar_pieza()
                    jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
                    qadr = self.model.jnt_qposadr[jid]
                    x0 = -1.05 + 0.18 * (i % 3)
                    self.data.qpos[qadr:qadr+3] = [x0, -self.LANE, OBJECT_REST_Z[i]]
                    self.data.qpos[qadr+3:qadr+7] = [1.0, 0.0, 0.0, 0.0]
                    self.data.qvel[dadr:dadr+6] = 0.0
                    self.segments[i] = 0
                    continue

            robot.control_robot_ik(self.model, self.data, self.robot_ids, self.robot_state, self.ROBOT_CONFIG)
            robot.follow_tcp_with_grasped_object(self.model, self.data, self.robot_ids, self.robot_state)

            mujoco.mj_step(self.model, self.data)
            
            self.lock_object_rotations()
            self.lock_cylinders_to_belt_center()
            robot.follow_tcp_with_grasped_object(self.model, self.data, self.robot_ids, self.robot_state)
            mujoco.mj_forward(self.model, self.data)

            self.horno.aumentar_temperatura(0.01)
            self.horno.actualizar_color(self.model)

    def procesar_teclado(self, key):
        if key == ord('p') or key == ord('P'):
            self.paused = not self.paused
            print("PAUSA GLOBAL" if self.paused else "SIMULACIÓN REANUDADA")
        elif ord('1') <= key <= ord('5'):
            idx = key - ord('1')
            activa = self.cintas[idx].conmutar()
            self.actualizar_actuadores_cintas()
            print(f"Cinta {idx+1}: {'ON' if activa else 'OFF'}")
        elif key == ord('r') or key == ord('R'):
            self.reset_objects()
            self.oven_counts[:] = [0, 0, 0, 0, 0]
            self.horno.numero_piezas = 0
            self.robot_state.clear()
            self.robot_state.update(robot.new_robot_state())
            robot.initialize_robot(self.model, self.data, self.robot_ids, self.robot_state)
            self.actualizar_actuadores_cintas()
            print("Planta y Panda reiniciados.")

    def renderizar_mosaico(self, sim_time, oven_total, robot_state):
        frames = []
        flags = [c.esta_activa() for c in self.cintas]
        flags_txt = " ".join(f"C{i+1}:{'ON' if v else 'OFF'}" for i, v in enumerate(flags))
        status = (f"t={sim_time:6.1f}s · {'PAUSA' if self.paused else 'EN MARCHA'} · "
                  f"{flags_txt} · Panda={robot_state['state']} · ciclos={robot_state['cycles']} · horno={oven_total}")
        
        for cam in self.camaras:
            if self.renderer and self.data:
                frame = cam.capturar(self.renderer, self.data, status_text=status)
                frames.append(frame)

        if len(frames) == 6:
            top = np.hstack((frames[0], frames[1], frames[2]))
            bottom = np.hstack((frames[3], frames[4], frames[5]))
            mosaic = np.vstack((top, bottom))
            
            cv2.putText(mosaic, "Q/ESC salir | P pausa | 1-5 cintas ON/OFF | R reset | S captura",
                        (12, mosaic.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2, cv2.LINE_AA)
            return mosaic
        return None

    def ejecutar(self):
        self.inicializar()
        
        overview_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
        viewer_ctx = mujoco.viewer.launch_passive(self.model, self.data)
        if viewer_ctx is not None:
            viewer_ctx.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer_ctx.cam.fixedcamid = overview_id

        print("\n=== PLANTA ROBOTIZADA: 4 CINTAS + PANDA + HORNO ===")
        print("Circuito: inferior -> derecha -> superior -> izquierda -> inferior")
        print("Estacion robot: Cinta 4 -> Franka Panda -> Cinta 5 -> horno")
        print(f"Piezas que recoge el Panda: {sorted(self.ROBOT_CONFIG['pick_object_ids'])}")
        print("Q/ESC: salir | P: pausa global | 1..5: ON/OFF cintas | R: reset | S: captura")
        print("Banderas iniciales:", " ".join(f"C{i+1}={'ON' if c.esta_activa() else 'OFF'}" for i, c in enumerate(self.cintas)))
        print(f"Camaras: FOV={CAM_FOVY:.0f} deg, altura={CAM_HEIGHT:.2f} m")
        print(f"Bloqueo de rotacion de objetos: {LOCK_OBJECT_ROTATION}")
        print(f"Cilindros centrados: {LOCK_CYLINDERS_TO_CENTERLINE}")
        print(f"Panda base=({self.ROBOT_CONFIG['base_x']:.2f}, {self.ROBOT_CONFIG['base_y']:.2f}, {self.ROBOT_CONFIG['base_z']:.2f})")
        print("Programacion orientada a objetos (OOP).\n")
        
        last_cam_time = -1.0
        try:
            while self.model and self.data:
                if viewer_ctx is not None and not viewer_ctx.is_running():
                    break

                self.actualizar()
                
                if viewer_ctx is not None:
                    viewer_ctx.sync()

                sim_time = self.data.time
                oven_total = self.horno.numero_piezas
                
                if self.renderer is not None:
                    if last_cam_time < 0.0 or (sim_time - last_cam_time) >= (1.0 / CAM_FPS) or self.paused:
                        mosaic = self.renderizar_mosaico(sim_time, oven_total, self.robot_state)
                        if mosaic is not None:
                            cv2.imshow("Camaras de la planta", mosaic)
                        last_cam_time = sim_time
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q') or key == ord('Q'):
                    break
                
                self.procesar_teclado(key)
        finally:
            if viewer_ctx is not None:
                viewer_ctx.close()
            if self.renderer is not None:
                self.renderer.close()
            cv2.destroyAllWindows()

    def construir_xml_planta(self):
        return build_mjcf()