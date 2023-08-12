# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hongyu Wen


import logging
import bpy 
import numpy as np
from math import sin, cos, pi, exp, sqrt

from infinigen.assets.creatures.util.creature import PartFactory, Part
from infinigen.assets.creatures.util.genome import Joint, IKParams
from infinigen.assets.creatures.util import part_util
from infinigen.core.util import blender as butil
from scipy.interpolate import interp1d

from infinigen.assets.creatures.util.geometry import nurbs as nurbs_util
from infinigen.core import surface

logger = logging.getLogger(__name__)

def square(x):
    return x * x
def sigmoid(x):
    return 1 / (1 + exp(-x))
def interpolate(coords):
    x = [c[0] for c in coords]
    y = [c[1] for c in coords]
    f = interp1d(x, y, kind='cubic')
    return f
def lr_scale(l, r, p):
    p = p * (r - l) + l
    return p
def lrlr_scale(l, r, L, R, p):
    p = (p - L) / (R - L)
    p = p * (r - l) + l
    return p
def sunk(l, r, gr, p):
    if p < l or p > r:
        return 1
    
    p = (p - l) / (r - l)
    return lrlr_scale(gr, 1, 0, 0.5, abs(p - 0.5))

def dist(x, y):
    return sqrt(square(x[0] - y[0]) + square(x[1] - y[1]))

def corner_vertices(obj):
    THRESHOLD = 0.3
    mesh = obj.data
    n = len(obj.data.vertices)
    value = np.zeros(n)

    nor = {}
    def add(u, v):
        if u not in nor:
            nor[u] = []
        nor[u].append(v)

    for face in mesh.polygons:
        for v in face.vertices:
            add(v, face.normal)

    COUNT = 0
    for u in mesh.vertices:
        if u.index not in nor:
            value[u.index] = 1.0
            COUNT += 1
            continue 

        normals = nor[u.index]
        mx_cross = 0
        for i in range(len(normals)):
            for j in range(i + 1, len(normals)):
                n1 = normals[i]
                n2 = normals[j]
                cross = np.linalg.norm(np.cross(n1, n2))
                mx_cross = max(mx_cross, cross)
                if mx_cross > THRESHOLD:
                    break
            if mx_cross > THRESHOLD:
                break
                
        if mx_cross > THRESHOLD:
            COUNT += 1
            value[u.index] = 1.0
    return value
def dorsal_vertices(obj):
    mesh = obj.data
    n = len(obj.data.vertices)
    value = np.zeros(n)

    for face in mesh.polygons:
        for v in face.vertices:
            if face.normal[2] > 0:
                value[v] = 1.0
    for u in mesh.vertices:
        if u.co[0] > 0:
            value[u.index] = 0
    return value
def ventral_vertices(obj, bodycheck=False):
    mesh = obj.data
    n = len(obj.data.vertices)
    value = np.zeros(n)

    for face in mesh.polygons:
        for v in face.vertices:
            if face.normal[2] < 0:
                value[v] = 1.0
    if bodycheck:
        for u in mesh.vertices:
            if u.co[0] > 0:
                value[u.index] = 0
    return value

class nurbs_ReptileTail():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.n = 20
        self.m = 50

    def local_scale(self, p):
        def foo(p):
            return 2 * p * (1 - p)
        
        sunken = self.sunken
        sunken *= sunk(-0.1, 0.1, 0.65, p)
        sunken *= sunk(self.wrist - 0.075, self.wrist + 0.075, 0.7, p)
        sunken = max(sunken, self.sunken_limit)

        if p < 0.4:
            p = lrlr_scale(0.3, self.body_curve, 0, 0.4, p)
        else:
            p = lrlr_scale(self.body_curve, 1, 0.4, 1, p)
        value = foo(p) * sunken
        return value


    def get_ctrls(self):
        self.n = int(self.n)
        self.m = int(self.m)
        ctrls = np.zeros((self.n, self.m, 3)) 
        for i in range(self.n):
            for j in range(self.m):
                p = i / (self.n - 1)
                theta = 2 * pi * j / (self.m)
                ctrls[i][j][0] = p
                ctrls[i][j][1] = self.local_scale(p) * cos(theta)
                ctrls[i][j][2] = self.local_scale(p) * sin(theta)
                
                ctrls[i][j][0] *= self.scale_x
                ctrls[i][j][1] *= self.scale_y
                ctrls[i][j][2] *= self.scale_z

                if (self.breast > 0.5):
                    if (0.10 < p and p < 0.25):
                        if abs(sin(theta)) < 0.5:
                            ctrls[i][j][1] *= 40 * (0.1 - abs(0.15 - p)) * (1 - abs(sin(theta)))
                        else:
                            ctrls[i][j][2] *= 0.8
                if (p > self.wrist):
                    ctrls[i][j][1] *= self.tail_modification
                    ctrls[i][j][2] *= self.tail_modification

        gp = int(3 * self.m / 4)
        ground = ctrls[0][gp][2]
        for i in range(self.n):
            float_ground = ctrls[i][gp][2]
            for j in range(self.m):
                ctrls[i][j][2] -= float_ground - ground

        return ctrls
    
    def generate(self):
        ctrls = self.get_ctrls()
        method = 'blender' if False else 'geomdl'
        return nurbs_util.nurbs(ctrls, method, face_size=0.1)

class nurbs_ReptileUpperHead():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.n = 50
        self.m = 50
        self.local_scale_y = self.init_local_scale_y()
        self.local_scale_z = self.init_local_scale_z()
        
    def local_offset_x(self, p):
        return -self.blunt_head * max(p - 0.9, 0)
    
    def local_offset_z(self, p, theta):
        a = sin(theta)
        if (a > 0.7):
            return -(a - 0.7) * 0.7
        return 0
    
    def init_local_scale_y(self):
        lrp = lambda p: lr_scale(-1/4, 1, p)
        return lambda p, theta: sqrt(1 - abs(lrp(p)) ** 1)
    
    def init_local_scale_z(self):
        def f1(p, theta):
            lrp = lambda p: lr_scale(-1/3, 1, p)
            return sqrt(1 - lrp(p) ** 2)
        def f2(p, theta):
            return (1 - p ** 2) / 10 
        
        def foo(p, theta):
            if 0 <= theta and theta <= pi:
                return 0.9 * f1(p, theta) + 0.1 * f2(p, theta)
            else:
                return 0.1 * f1(p, theta) + 0.9 * f2(p, theta)
        return foo
    
    def update(self, set):
        xs, ys, k = set
        for x in xs:
            for y in ys:
                self.ctrls[x][y][1] *= k[0]
                self.ctrls[x][y][2] *= k[1]

    def bump(self, pos, degree, boundx, boundy):
        cx = round(self.n * pos[0])
        cy = round(self.m * pos[1])
        lx = round(self.n * boundx[0])
        rx = round(self.m * boundx[1])
        ly = round(self.n * boundy[0])
        ry = round(self.m * boundy[1])
        for i in range(cx - lx, cx + rx + 1):
            for j in range(cy - ly, cy + ry + 1):
                self.ctrls[i][j][2] *= 1 + max(0, degree * lrlr_scale(0, 1, 5 * 1.4, 0, dist((cx, cy), (i, j))))


    def get_ctrls(self):
        self.n = int(self.n)
        self.m = int(self.m)
        ctrls = np.zeros((self.n, self.m, 3)) 
        for i in range(self.n):
            for j in range(self.m):
                p = i / (self.n - 1)
                theta = 2 * pi * j / (self.m)
                ctrls[i][j][0] = p + self.local_offset_x(p)
                ctrls[i][j][1] = self.local_scale_y(p, theta) * cos(theta)
                ctrls[i][j][2] = self.local_scale_z(p, theta) * (sin(theta) + self.local_offset_z(p, theta))
                
                ctrls[i][j][0] *= self.scale_x
                ctrls[i][j][1] *= self.scale_y
                ctrls[i][j][2] *= self.scale_z
                
                
        self.ctrls = ctrls

        # snakes
        teeth_range = [31, 32, 33, 34, 41, 42, 43, 44]
        settings = [
            # ([42], [4, 21], (0.7, 1)),
            ([32, 33, 34], [5, 20], (0.6, 1)),
            (range(31, 43), [6, 7, 18, 19], (1.02, 1.06)),
            ([30], [6, 7, 18, 19], (1, 1.05)),
            ([29], [6, 7, 18, 19], (1, 1.03)),
            (range(25, 29), [6, 7, 18, 19], (1, 1.01)),
            (range(15, 22), teeth_range, (1, 1.5)),
            ([22], teeth_range, (1, 2)),
            ([23], teeth_range, (1, 2.2)),
            ([24], teeth_range, (1, 2.5)),
            (range(25, 40), teeth_range, (1, 2)),
            ([32], teeth_range, (1, 1.3)),
            ([33], teeth_range, (1, 1.5)),
            ([34], teeth_range, (1, 1.8)),
            (range(35, 40), teeth_range, (1, 2)),
        ]

        # frog
        # eye_socket_range = [6, 7, 8, 17, 18, 19]
        # eye_socket_range_2 = [9, 10, 15, 16]
        # settings = [
        #     ([32, 33, 34], [5, 20], (0.6, 1)), # eye
        #     ([38, 39], eye_socket_range, (1.02, 1.2)),
        #     (range(31, 38), eye_socket_range, (1.05, 1.3)),
        #     ([30], eye_socket_range, (1.04, 1.2)),
        #     ([29], eye_socket_range, (1.02, 1.1)),
        #     ([28], eye_socket_range, (1.01, 1.05)),
        #     ([38, 39], eye_socket_range_2, (1.02, 1.15)),
        #     (range(31, 38), eye_socket_range_2, (1.05, 1.2)),
        #     ([30], eye_socket_range_2, (1.04, 1.15)),
        #     ([29], eye_socket_range_2, (1.02, 1.08)),
        #     ([28], eye_socket_range_2, (1.01, 1.04)),
        # ]
        for set in settings:
            self.update(set)
            
        for i in range(self.n):
            for j in range(self.m):
                p = i / (self.n - 1)
                theta = 2 * pi * j / (self.m)
                if (p >= self.up_head_position):
                    self.ctrls[i][j][2] += self.up_head_degree * (p - self.up_head_position)
                if (sin(theta) < -0.6):
                    self.ctrls[i][j][2] += 0.3 * (-sin(theta) - 0.6)
        # self.bump((0.8, 0.2), 0.2, (0.15, 0.15), (0.15, 0.02))
        for i in range(self.n):
            for j in range(self.m):
                self.ctrls[i][j][0] += self.offset_x
                self.ctrls[i][j][1] += self.offset_y
                self.ctrls[i][j][2] += self.offset_z
                
        return self.ctrls
    
    def generate(self):
        ctrls = self.get_ctrls()
        method = 'blender' if False else 'geomdl'
        obj = nurbs_util.nurbs(ctrls, method, face_size=0.05)
        surface.new_attr_data(obj, 'corner', 'FLOAT', 'POINT', corner_vertices(obj))
        surface.new_attr_data(obj, 'inside_mouth', 'FLOAT', 'POINT', ventral_vertices(obj))
        return obj

class nurbs_ReptileLowerHead():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.n = 50
        self.m = 50
        self.local_scale_y = self.init_local_scale_y()
        self.local_scale_z = self.init_local_scale_z()
        
    def local_offset_x(self, p):
        return -0.7 * max(p - 0.9, 0)
    
    def local_offset_z(self, p, theta):
        a = sin(theta)
        if (a > 0.7):
            return -(a - 0.7) ** 2 * 0.7
        return 0
    
    def init_local_scale_y(self):
        lrp = lambda p: lr_scale(-1/4, 1, p)
        return lambda p, theta: sqrt(1 - abs(lrp(p)) ** 1)
    
    def init_local_scale_z(self):
        def foo(p, theta):
            if 0 <= theta and theta <= pi:
                return sqrt(1 - p ** 2)
            else:
                return (1 - p ** 2) / 10 
        return foo
    
    def update(self, set):
        xs, ys, k = set
        for x in xs:
            for y in ys:
                self.ctrls[x][y][1] += k[0]
                self.ctrls[x][y][2] += k[1]

    def get_ctrls(self):
        self.n = int(self.n)
        self.m = int(self.m)
        ctrls = np.zeros((self.n, self.m, 3)) 
        for i in range(self.n):
            for j in range(self.m):
                p = i / (self.n - 1)
                theta = 2 * pi * j / (self.m)
                ctrls[i][j][0] = p + self.local_offset_x(p)
                ctrls[i][j][1] = self.local_scale_y(p, theta) * cos(theta)
                ctrls[i][j][2] = self.local_scale_z(p, theta) * (sin(theta) + self.local_offset_z(p, theta))
                
                ctrls[i][j][0] *= self.scale_x
                ctrls[i][j][1] *= self.scale_y
                ctrls[i][j][2] *= -self.scale_z
                
                if (p > 0.6):
                    ctrls[i][j][2] += 0.4 * ((p - 0.6) ** 2)
                
        self.ctrls = ctrls
        for i in range(self.n):
            for j in range(self.m):
                self.ctrls[i][j][0] += self.offset_x
                self.ctrls[i][j][1] += self.offset_y
                self.ctrls[i][j][2] += self.offset_z
        return self.ctrls
    
    def generate(self):
        ctrls = self.get_ctrls()
        method = 'blender' if False else 'geomdl'
        obj = nurbs_util.nurbs(ctrls, method, face_size=0.01)
        surface.new_attr_data(obj, 'corner', 'FLOAT', 'POINT', corner_vertices(obj))
        surface.new_attr_data(obj, 'inside_mouth', 'FLOAT', 'POINT', ventral_vertices(obj))
        return obj

class nurbs_ReptileHead():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.n = 50
        self.m = 50
        self.local_scale_y = self.init_local_scale_y()
        self.local_scale_z = self.init_local_scale_z()
        
    def local_offset_x(self, p):
        return -self.blunt_head * max(p - 0.9, 0)
    
    def local_offset_z(self, p, theta):
        a = sin(theta)
        if (a > 0.7):
            return -(a - 0.7) * 0.7
        return 0
    
    def init_local_scale_y(self):
        lrp = lambda p: lr_scale(-1/2, 1, p)
        return lambda p, theta: sqrt(1 - abs(lrp(p)) ** 1)
    
    def init_local_scale_z(self):
        def f1(p, theta):
            lrp = lambda p: lr_scale(-1/3, 1, p)
            return sqrt(1 - lrp(p) ** 2)
        def f2(p, theta):
            return (1 - p ** 2) / 10 
        
        def foo(p, theta):
            if 0 <= theta and theta <= pi:
                return 0.9 * f1(p, theta) + 0.1 * f2(p, theta)
            else:
                return 0.8 * f1(p, theta) + 0.2 * f2(p, theta)
        return foo
    
    def update(self, set):
        xs, ys, k = set
        for x in xs:
            for y in ys:
                self.ctrls[x][y][1] *= k[0]
                self.ctrls[x][y][2] *= k[1]

    def update_all(self, scale, offset=(0, 0, 0)):
        l, r = scale
        
        n, m, _ = self.ctrls.shape
        for i in range(n):
            for j in range(m):
                self.ctrls[i][j][0] *= (r - l)
                self.ctrls[i][j][1] *= abs(r - l)
                self.ctrls[i][j][2] *= abs(r - l)
                
                self.ctrls[i][j][0] += offset[0]
                self.ctrls[i][j][1] += offset[1]
                self.ctrls[i][j][2] += offset[2]
        if (l > r):
            self.ctrls = np.flip(self.ctrls, axis=0)
        return self.ctrls

    def get_ctrls(self):
        self.n = int(self.n)
        self.m = int(self.m)
        ctrls = np.zeros((self.n, self.m, 3)) 
        for i in range(self.n):
            for j in range(self.m):
                p = i / (self.n - 1)
                theta = 2 * pi * j / (self.m)
                ctrls[i][j][0] = p + self.local_offset_x(p)
                ctrls[i][j][1] = self.local_scale_y(p, theta) * cos(theta)
                ctrls[i][j][2] = self.local_scale_z(p, theta) * (sin(theta) + self.local_offset_z(p, theta))
                
                ctrls[i][j][0] *= self.scale_x
                ctrls[i][j][1] *= self.scale_y
                ctrls[i][j][2] *= self.scale_z
                
                
        self.ctrls = ctrls
        settings = [
            ([42], [4, 21], (0.7, 1)),
            ([32, 33, 34], [5, 20], (0.6, 1)),
            (range(31, 43), [6, 7, 18, 19], (1.02, 1.06)),
            ([30], [6, 7, 18, 19], (1, 1.05)),
            ([29], [6, 7, 18, 19], (1, 1.03)),
        ]
        for set in settings:
            self.update(set)

        for i in range(self.n):
            for j in range(self.m):
                p = i / (self.n - 1)
                theta = 2 * pi * j / (self.m)
                if (p >= self.up_head_position):
                    self.ctrls[i][j][2] += self.up_head_degree * (p - self.up_head_position)
                if (sin(theta) < -0.6):
                    self.ctrls[i][j][2] += 0.3 * (-sin(theta) - 0.6)
        
        # self.update_all((1, 0), (0, 0, -0.05))
        return self.ctrls
    
    def generate(self):
        ctrls = self.get_ctrls()
        method = 'blender' if False else 'geomdl'
        return nurbs_util.nurbs(ctrls, method, face_size=0.05)

class nurbs_LizardFrontLeg():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.n = 50
        self.m = 50
        self.local_scale_y = self.init_local_scale_y()
        self.local_scale_z = self.init_local_scale_z()
        self.local_offset_x = self.init_local_offset_x()
        self.local_offset_y = self.init_local_offset_y()
        self.scale_x *= 0.6
        self.scale_y *= 0.4
        self.scale_z *= 0.4

        
    def init_local_offset_x(self):
        return lambda p, theta: 0
        alpha = pi * 0.45
        thred = 0.9
        def foo(p, theta):
            offset = 0
            if (p <= thred):
                offset += -(1 - cos(alpha)) * (p - thred)
            offset -= -(1 - cos(alpha)) * (0 - thred)
            return offset
        return foo
    
    def init_local_offset_y(self):
        return lambda p, theta: 0
        alpha = -pi * 0.45
        thred = 0.9
        def foo(p, theta):
            offset = 0
            if (p <= thred):
                offset += sin(alpha) * (p - thred)
            # offset -= sin(alpha) * (0 - thred)
            return offset
        return foo
    
    def init_local_scale_y(self):
        th0 = 0.03
        th1 = 0.1
        th2 = 0.7

        cr1 = -3
        cr2 = 0.8
        cr3 = 1.4

        bar1 = 0.3 - (th0 - th1) * sin(lrlr_scale(0, pi, th0, th1, th1 - th0))
        bar2 = 0.3 - cr2 * th1
        bar3 = cr2 * th2 + bar2 - cr3 * th2
        
        def foo(p, theta):
            p = 1 - p
            if (p < th0):
                return cos(lrlr_scale(0, pi / 2, 0, th0, th0 - p))
            elif (p < th1):
                return (th0 - th1) * sin(lrlr_scale(0, pi, th0, th1, p - th0)) + bar1
            elif (p < th2):
                return cr2 * p + bar2
            else:
                return cr3 * p + bar3
        return foo
    
    def init_local_scale_z(self):
        def foo(p, theta):
            return 0.2 + 0.8 * (1 - p)
        return foo
    
    def update(self, set):
        xs, ys, k = set
        for x in xs:
            for y in ys:
                self.ctrls[x][y][1] *= k[0]
                self.ctrls[x][y][2] *= k[1]

    def get_ctrls(self):
        self.n = int(self.n)
        self.m = int(self.m)
        ctrls = np.zeros((self.n, self.m, 3)) 
        for i in range(self.n):
            for j in range(self.m):
                p = i / (self.n - 1)
                theta = 2 * pi * j / (self.m)
                ctrls[i][j][0] = p + self.local_offset_x(p, theta)
                ctrls[i][j][1] = self.local_scale_y(p, theta) * (cos(theta) + self.local_offset_y(p, theta))
                ctrls[i][j][2] = self.local_scale_z(p, theta) * sin(theta)
                
                ctrls[i][j][0] *= self.scale_x
                ctrls[i][j][1] *= self.scale_y
                ctrls[i][j][2] *= self.scale_z

        self.ctrls = ctrls        
        return self.ctrls
    
    def generate(self):
        ctrls = self.get_ctrls()
        method = 'blender' if False else 'geomdl'
        return nurbs_util.nurbs(ctrls, method, face_size=0.015)

class nurbs_LizardBackLeg():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.n = 50
        self.m = 50
        self.local_scale_y = self.init_local_scale_y()
        self.local_scale_z = self.init_local_scale_z()
        self.local_offset_x = self.init_local_offset_x()
        self.local_offset_y = self.init_local_offset_y()
        self.scale_x *= 0.6
        self.scale_y *= 0.4
        self.scale_z *= 0.4
        
    def init_local_offset_x(self):
        return lambda p, theta: 0
        alpha = -pi * 0.45
        thred = 0.9
        def foo(p, theta):
            offset = 0
            if (p <= thred):
                offset += -(1 - cos(alpha)) * (p - thred)
            offset -= -(1 - cos(alpha)) * (0 - thred)
            return offset
        return foo
    
    def init_local_offset_y(self):
        return lambda p, theta: 0
        alpha = pi * 0.45
        thred = 0.9
        def foo(p, theta):
            offset = 0
            if (p <= thred):
                offset += sin(alpha) * (p - thred)
            # offset -= sin(alpha) * (0 - thred)
            return offset
        return foo
    
    def init_local_scale_y(self):
        th0 = 0.03
        th1 = 0.1
        th2 = 0.7

        cr1 = -3
        cr2 = 0.8
        cr3 = 1.4

        bar1 = 0.3 - (th0 - th1) * sin(lrlr_scale(0, pi, th0, th1, th1 - th0))
        bar2 = 0.3 - cr2 * th1
        bar3 = cr2 * th2 + bar2 - cr3 * th2
        
        def foo(p, theta):
            p = 1 - p
            if (p < th0):
                return cos(lrlr_scale(0, pi / 2, 0, th0, th0 - p))
            elif (p < th1):
                return (th0 - th1) * sin(lrlr_scale(0, pi, th0, th1, p - th0)) + bar1
            elif (p < th2):
                return cr2 * p + bar2
            else:
                return cr3 * p + bar3
        return foo
    
    def init_local_scale_z(self):
        def foo(p, theta):
            return 0.2 + 0.8 * (1 - p)
        return foo
    
    def update(self, set):
        xs, ys, k = set
        for x in xs:
            for y in ys:
                self.ctrls[x][y][1] *= k[0]
                self.ctrls[x][y][2] *= k[1]

    def get_ctrls(self):
        self.n = int(self.n)
        self.m = int(self.m)
        ctrls = np.zeros((self.n, self.m, 3)) 
        for i in range(self.n):
            for j in range(self.m):
                p = i / (self.n - 1)
                theta = 2 * pi * j / (self.m)
                ctrls[i][j][0] = p + self.local_offset_x(p, theta)
                ctrls[i][j][1] = self.local_scale_y(p, theta) * (cos(theta) + self.local_offset_y(p, theta))
                ctrls[i][j][2] = self.local_scale_z(p, theta) * sin(theta)
                
                ctrls[i][j][0] *= self.scale_x
                ctrls[i][j][1] *= self.scale_y
                ctrls[i][j][2] *= self.scale_z

        self.ctrls = ctrls        
        return self.ctrls
    
    def generate(self):
        ctrls = self.get_ctrls()
        method = 'blender' if False else 'geomdl'
        return nurbs_util.nurbs(ctrls, method, face_size=0.015)

class nurbs_LizardToe():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.n = 50
        self.m = 50
        self.scale_x = 0.15
        self.scale_y = 0.035
        self.scale_z = 0.035
        self.local_scale_y = self.init_local_scale()
        self.local_scale_z = self.init_local_scale()
        self.local_offset_x = self.init_local_offset_x()
        self.local_offset_y = self.init_local_offset_y()
        self.scale_x *= 0.6
        self.scale_y *= 0.4
        self.scale_z *= 0.4
        
    def init_local_offset_x(self):
        def foo(p, theta):
            if (p < 0.98):
                return 0
            else:
                return (p - 0.9) * 3
        return foo
    
    def init_local_offset_y(self):
        return lambda p, theta: 0

    def init_local_scale(self):
        def foo(p, theta):
            if (p < 0.4):
                return 1
            elif (p < 0.9):
                return 1 + 0.5 * sin(lrlr_scale(0, pi, 0, 0.5, p - 0.4))
            else:
                return cos(lrlr_scale(0, pi / 2, 0, 0.1, p - 0.9))
        return foo
    
    def update(self, set):
        xs, ys, k = set
        for x in xs:
            for y in ys:
                self.ctrls[x][y][1] *= k[0]
                self.ctrls[x][y][2] *= k[1]

    def get_ctrls(self):
        self.n = int(self.n)
        self.m = int(self.m)
        ctrls = np.zeros((self.n, self.m, 3)) 
        for i in range(self.n):
            for j in range(self.m):
                p = i / (self.n - 1)
                theta = 2 * pi * j / (self.m)
                ctrls[i][j][0] = p + self.local_offset_x(p, theta)
                ctrls[i][j][1] = self.local_scale_y(p, theta) * (cos(theta) + self.local_offset_y(p, theta))
                ctrls[i][j][2] = self.local_scale_z(p, theta) * sin(theta)
                
                ctrls[i][j][0] *= self.scale_x
                ctrls[i][j][1] *= self.scale_y
                ctrls[i][j][2] *= self.scale_z

        self.ctrls = ctrls        
        return self.ctrls
    
    def generate(self):
        ctrls = self.get_ctrls()
        method = 'blender' if False else 'geomdl'
        return nurbs_util.nurbs(ctrls, method, face_size=0.005)

class nurbs_ReptileBody():
    def __init__(self, head, tail, **kwargs):
        self.__dict__.update(kwargs)
        self.param_head = head
        self.param_tail = tail
        self.m = 50
         
    def position(self, x):
        return np.array([x, 0, sigmoid((x - 0.5) * 4)])
         
    def update(self, ctrls, scale, offset=(0, 0, 0)):
        l, r = scale
        
        n, m, _ = ctrls.shape
        for i in range(n):
            for j in range(m):
                ctrls[i][j][0] *= (r - l)
                ctrls[i][j][1] *= abs(r - l)
                ctrls[i][j][2] *= abs(r - l)
                
                ctrls[i][j][0] += offset[0]
                ctrls[i][j][1] += offset[1]
                ctrls[i][j][2] += offset[2]
        if (l > r):
            ctrls = np.flip(ctrls, axis=0)
        return ctrls
    
    def merge(self, c1, c2):
        nc1 = np.copy(c1)
        nc2 = np.copy(c2)
        nc2[0] = 0.5 * c2[0] + 0.5 * c1[-1]
        nc1[-1] = nc2[0] 
        for i in range(2, 6):
            nc1[-i] = 0.1 * ((i + 4) * c1[-i] + (6 - i) * c2[0])
        for i in range(1, 5):
            nc2[i] = 0.1 * ((i + 5) * c2[i] + (5 - i) * c1[-1])
        return np.concatenate((nc1, nc2), axis=0)
    
    def get_ctrls(self):
        if self.open_mouth:
            head = nurbs_ReptileUpperHead(**self.param_head)
        else:
            head = nurbs_ReptileHead(**self.param_head)
        tail = nurbs_ReptileTail(**self.param_tail)
        
        head_ctrls = head.get_ctrls()
        tail_ctrls = tail.get_ctrls()
        
        if self.open_mouth:
            head_ctrls = self.update(head_ctrls, (1, 0), (0, 0, -0.05))
        else:
            head_ctrls = self.update(head_ctrls, (1, 0), (0, 0, -0.05))
        tail_ctrls = self.update(tail_ctrls, (0, 1))
        self.ctrls = self.merge(head_ctrls, tail_ctrls)
        return self.ctrls
    
    def generate(self):
        ctrls = self.get_ctrls()
        method = 'blender' if False else 'geomdl'
        return nurbs_util.nurbs(ctrls, method, face_size=0.5)
         
class ReptileHeadBody(PartFactory):
    param_templates = {}
    tags = ['body']
    unit_scale = (0.5, 0.5, 0.5)

    def __init__(self, params=None, type='lizard'):
        self.type = type
        super().__init__(params)

    def sample_params(self, select=None, var=1):
        params = self.param_templates[self.type]
        # weights = part_util.random_convex_coord(param_templates.keys(), select=select)
        # params = part_util.rdict_comb(param_templates, weights)
        
        N = lambda m, v: np.random.normal(m, v * var)
        U = lambda l, r: np.random.uniform(l, r)

        for key in params['tail']:
            l, r = params['trange'][key]
            noise = N(0, 0.1 * (r - l))
            params['tail'][key] += noise
        for key in params['head']:
            l, r = params['hrange'][key]
            noise = N(0, 0.1 * (r - l))
            params['head'][key] += noise
        
        return params
    
    def rescale(self, params, scale):
        params['sx'] *= scale
        params['sy'] *= scale
        params['sz'] *= scale
        return params

    def make_part(self, params):
        handles = nurbs_ReptileBody(**params).get_ctrls()
        part = part_util.nurbs_to_part(handles)
        part.skeleton = handles.mean(axis=1)
        part.joints = {
            0.1: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]])), # head
            0.73: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]])), # neck
            0.80: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]])), # waist
            0.85: Joint(rest=(0,0,0), bounds=np.array([[-30, 0, -30], [30, 0, 30]])),
            0.88: Joint(rest=(0,0,0), bounds=np.array([[-30, 0, -30], [30, 0, 30]])),
            0.92: Joint(rest=(0,0,0), bounds=np.array([[-30, 0, -30], [30, 0, 30]])),
            0.95: Joint(rest=(0,0,0), bounds=np.array([[-30, 0, -30], [30, 0, 30]])),
            0.98: Joint(rest=(0,0,0), bounds=np.array([[-30, 0, -30], [30, 0, 30]])),
            1.0: Joint(rest=(0,0,0), bounds=np.array([[-30, 0, -30], [30, 0, 30]])),
        }
        if self.type == 'snake':
            part.iks = {
                0.1: IKParams('snake_head', rotation_weight=0.1, chain_parts=2),
                0.73: IKParams(name='snake_shoulder', rotation_weight=0.1, target_size=0.4),
                0.85: IKParams(name='snake_hip', target_size=0.3),
                1.0: IKParams(name='snake_tail', chain_parts=1)
        }
        else:
            part.iks = {
                0.1: IKParams('head', rotation_weight=0.1, chain_parts=2),
                0.73: IKParams(name='shoulder', rotation_weight=0.1, target_size=0.4),
                0.85: IKParams(name='hip', target_size=0.3),
                1.0: IKParams(name='reptile_tail', chain_parts=1)
        }
        surface.new_attr_data(part.obj, 'corner', 'FLOAT', 'POINT', corner_vertices(part.obj))
        surface.new_attr_data(part.obj, 'inside_mouth', 'FLOAT', 'POINT', ventral_vertices(part.obj, bodycheck=True))
        return part

class ReptileBody(PartFactory):
    param_templates = {}
    tags = ['body']
    unit_scale = (0.5, 0.5, 0.5)

    def __init__(self, params=None, type='lizard', n_bones=None, shoulder_ik_ts=None, mod=None):
        self.type = type
        self.n_bones = n_bones
        self.shoulder_ik_ts = shoulder_ik_ts
        self.mod = mod
        super().__init__(params)

    def sample_params(self, select=None, var=1):
        params = self.param_templates[self.type]

        N = lambda m, v: np.random.normal(m, v * var)
        U = lambda l, r: np.random.uniform(l, r)

        for key in params['tail']:
            l, r = params['trange'][key]
            noise = N(0, 0.1 * (r - l))
            params['tail'][key] += noise
        
        return params['tail']
    
    def rescale(self, handles):
        if self.mod is None:
            return handles
        
        handles[:,:,0] *= self.mod[0]
        handles[:,:,1] *= self.mod[1]
        handles[:,:,2] *= self.mod[2]
        return handles

    def make_part(self, params):
        logger.debug(params)
        handles = nurbs_ReptileTail(**params).get_ctrls()
        handles = self.rescale(handles)
        
        part = part_util.nurbs_to_part(handles)
        part.skeleton = handles.mean(axis=1)
        part.joints = {
            i: Joint((0,0,0), bounds=np.array([[-30, -30, -30], [30, 30, 30]]))
            for i in np.linspace(0, 1, self.n_bones, endpoint=True)
        }
        part.iks = {
            t: IKParams(name=f'body_{i+1}', mode='pin' if i == 0 else 'iksolve', 
                        rotation_weight=0, target_size=0.3)
            for i, t in enumerate(self.shoulder_ik_ts)
        }
        surface.new_attr_data(part.obj, 'corner', 'FLOAT', 'POINT', corner_vertices(part.obj))
        surface.new_attr_data(part.obj, 'inside_mouth', 'FLOAT', 'POINT', ventral_vertices(part.obj, bodycheck=True))
        return part

class ReptileUpperHead(PartFactory):
    param_templates = {}
    tags = ['jaw']

    def __init__(self, params=None, mod=None):
        self.mod = mod
        super().__init__(params)

    def sample_params(self, select=None, var=1):
        params = self.param_templates
        # weights = part_util.random_convex_coord(self.param_templates.keys(), select=select)
        # params = part_util.rdict_comb(self.param_templates, weights)
        # params = np.random.choice(list(self.param_templates.values()))
        
        N = lambda m, v: np.random.normal(m, v * var)
        U = lambda l, r: np.random.uniform(l, r)
        for key in params['head']:
            if key in params['range']:
                l, r = params['range'][key]
                noise = N(0, 0.1 * (r - l))
                params['head'][key] += noise
        return params['head']
    
    def rescale(self, handles):
        if self.mod is None:
            return handles
        
        handles[:,:,0] *= self.mod[0]
        handles[:,:,1] *= self.mod[1]
        handles[:,:,2] *= self.mod[2]
        return handles

    def make_part(self, params):
        logger.debug(params)
        handles = nurbs_ReptileUpperHead(**params).get_ctrls()
        handles = self.rescale(handles)
        
        part = part_util.nurbs_to_part(handles, 0.01)
        part.skeleton = handles.mean(axis=1)
        # part.iks = {1.0: IKParams('body_0', rotation_weight=0.1, chain_parts=1)}
        surface.new_attr_data(part.obj, 'corner', 'FLOAT', 'POINT', corner_vertices(part.obj))
        surface.new_attr_data(part.obj, 'inside_mouth', 'FLOAT', 'POINT', ventral_vertices(part.obj))
        return part

class ReptileLowerHead(PartFactory):
    param_templates = {}
    tags = ['jaw']

    def __init__(self, params=None, mod=None):
        self.mod = mod
        super().__init__(params)

    def sample_params(self, select=None, var=1):
        params = self.param_templates
        # weights = part_util.random_convex_coord(self.param_templates.keys(), select=select)
        # params = part_util.rdict_comb(self.param_templates, weights)
        # params = np.random.choice(list(self.param_templates.values()))
        
        N = lambda m, v: np.random.normal(m, v * var)
        U = lambda l, r: np.random.uniform(l, r)
        for key in params['head']:
            if key in params['range']:
                l, r = params['range'][key]
                noise = N(0, 0.1 * (r - l))
                params['head'][key] += noise
        return params['head']
    
    def rescale(self, handles):
        if self.mod is None:
            return handles
        
        handles[:,:,0] *= self.mod[0]
        handles[:,:,1] *= self.mod[1]
        handles[:,:,2] *= self.mod[2]
        return handles

    def make_part(self, params):
        handles = nurbs_ReptileLowerHead(**params).get_ctrls()
        handles = self.rescale(handles)
        
        part = part_util.nurbs_to_part(handles, 0.015)
        part.skeleton = handles.mean(axis=1)
        # part.iks = {1.0: IKParams('body_0', rotation_weight=0.1, chain_parts=1)}
        surface.new_attr_data(part.obj, 'corner', 'FLOAT', 'POINT', corner_vertices(part.obj))
        surface.new_attr_data(part.obj, 'inside_mouth', 'FLOAT', 'POINT', ventral_vertices(part.obj))
        return part

class LizardFrontLeg(PartFactory):
    param_templates = {}
    tags = ['leg']
    

    def __init__(self, params=None, type='lizard'):
        self.type = type
        super().__init__(params)

    def sample_params(self, select=None, var=1):
        params = self.param_templates[self.type]
        # weights = part_util.random_convex_coord(self.param_templates.keys(), select=select)
        # params = part_util.rdict_comb(self.param_templates, weights)
        # params = np.random.choice(list(self.param_templates.values()))
        
        N = lambda m, v: np.random.normal(m, v * var)
        U = lambda l, r: np.random.uniform(l, r)

        for key in params['leg']:
            l, r = params['range'][key]
            noise = N(0, 0.1 * (r - l))
            params['leg'][key] += noise

        return params['leg']
    
    def rescale(self, params, scale):
        params['sx'] *= scale
        params['sy'] *= scale
        params['sz'] *= scale
        return params

    def make_part(self, params):
        handles = nurbs_LizardFrontLeg(**params).get_ctrls()
        part = part_util.nurbs_to_part(handles, 0.015)
        part.skeleton = handles.mean(axis=1)
        part.joints = {
            0: Joint(rest=(0,0,0), bounds=np.array([[-30, 0, -30], [100, 0, 100]])), # shoulder
            0.4: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]])), # knee
            0.9: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]])), # ankle
        } 
        # part.obj.scale = (0.5, 0.5, 0.5)
        # butil.apply_transform(part.obj, scale=True)
        # part.iks = {
        #     0.1: IKParams('knee', rotation_weight=0.1, chain_parts=1),
        #     # 0.9: IKParams('ankle', rotation_weight=0.1, chain_parts=1),
        #     1.0: IKParams('foot', rotation_weight=0.1, chain_parts=1)}
        return part

class LizardBackLeg(PartFactory):
    param_templates = {}
    tags = ['leg']

    def __init__(self, params=None, type='lizard'):
        self.type = type
        super().__init__(params)

    def sample_params(self, select=None, var=1):
        params = self.param_templates[self.type]
        # weights = part_util.random_convex_coord(self.param_templates.keys(), select=select)
        # params = part_util.rdict_comb(self.param_templates, weights)
        # params = np.random.choice(list(self.param_templates.values()))
        
        N = lambda m, v: np.random.normal(m, v * var)
        U = lambda l, r: np.random.uniform(l, r)

        for key in params['leg']:
            l, r = params['range'][key]
            noise = N(0, 0.1 * (r - l))
            params['leg'][key] += noise

        return params['leg']

    def rescale(self, params, scale):
        params['sx'] *= scale
        params['sy'] *= scale
        params['sz'] *= scale
        return params

    def make_part(self, params):
        handles = nurbs_LizardBackLeg(**params).get_ctrls()
        part = part_util.nurbs_to_part(handles, 0.015)
        part.skeleton = handles.mean(axis=1)
        part.joints = {
            0: Joint(rest=(0,0,0), bounds=np.array([[-30, 0, -30], [100, 0, 100]])), # shoulder
            0.4: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]])), # knee
            0.9: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]])), # ankle
        } 
        # part.iks = {
        #             0.1: IKParams('knee', rotation_weight=0.1, chain_parts=1),
        #             # 0.9: IKParams('ankle', rotation_weight=0.1, chain_parts=1),
        #             1.0: IKParams('foot', rotation_weight=0.1, chain_parts=1)}
        return part

class LizardToe(PartFactory):
    param_templates = {}
    tags = ['foot_detail']

    def sample_params(self, select=None, var=1):
        weights = part_util.random_convex_coord(self.param_templates.keys(), select=select)
        params = part_util.rdict_comb(self.param_templates, weights)
        # params = np.random.choice(list(self.param_templates.values()))
        
        N = lambda m, v: np.random.normal(m, v * var)
        U = lambda l, r: np.random.uniform(l, r)
        return params
    
    def rescale(self, params, scale):
        params['sx'] *= scale
        params['sy'] *= scale
        params['sz'] *= scale
        return params

    def make_part(self, params):
        handles = nurbs_LizardToe(**params).get_ctrls()
        part = part_util.nurbs_to_part(handles, 0.005)
        # part.obj.scale = (0.5, 0.5, 0.5)
        # butil.apply_transform(part.obj, scale=True)

        part.joints = {
            0: Joint(rest=(0,0,0), bounds=np.array([[-35, 0, -70], [35, 0, 70]])),
        } 
        return part


lizard_tail = {
    'scale_x': 7,
    'scale_y': 0.65,
    'scale_z': 0.6,
    'sunken': 1,
    'sunken_limit': 0.9,
    'breast': 0,
    'body_curve': 0.75,
    'wrist': 0.32,
    'tail_modification': 0.8
}

lizard_tail_range = {
    'scale_x': [4, 10],
    'scale_y': [0.5, 1.5],
    'scale_z': [0.8, 1.5],
    'sunken': [0.8, 1.2],
    'sunken_limit': [0.5, 0.7],
    'breast': [0, 0],
    'body_curve': [0.7, 0.8],
    'wrist': [0.50, 0.55],
    'tail_modification': [0.3, 0.7]
}

for k, v in lizard_tail_range.items():
    lizard_tail_range[k] = np.array(v)

dinosaur_tail = {
    'scale_x': 30,
    'scale_y': 3,
    'scale_z': 3,
    'sunken': 1,
    'sunken_limit': 1,
    'breast': 0,
    'body_curve': 0.8,
    'wrist': 0.4,
    'tail_modification': 0.4
}

dinosaur_tail_range = {
    'scale_x': [4, 40],
    'scale_y': [5, 5],
    'scale_z': [5, 5],
    'sunken': [0.8, 1.2],
    'sunken_limit': [0.5, 0.7],
    'breast': [0, 0],
    'body_curve': [0.75, 0.75],
    'wrist': [0.50, 0.55],
    'tail_modification': [0.3, 0.7]
}

for k, v in dinosaur_tail_range.items():
    dinosaur_tail_range[k] = np.array(v)


snake_tail = {
    'scale_x': 30,
    'scale_y': 0.65,
    'scale_z': 0.57,
    'sunken': 1,
    'sunken_limit': 1,
    'breast': 0.5,
    'body_curve': 0.75,
    'wrist': 0.52,
    'tail_modification': 1,
}

snake_tail_range = {
    'scale_x': [20, 40],
    'scale_y': [0, 0],
    'scale_z': [0, 0],
    'sunken': [1, 1],
    'sunken_limit': [1, 1],
    'breast': [0, 1],
    'body_curve': [0.7, 0.8],
    'wrist': [0.50, 0.55],
    'tail_modification': [1, 1],
}
for k, v in snake_tail_range.items():
    snake_tail_range[k] = np.array(v)


frog_tail = {
    'scale_x': 2,
    'scale_y': 0.8,
    'scale_z': 0.8,
    'sunken': 1,
    'sunken_limit': 1,
    'breast': 0,
    'body_curve': 0.75,
    'wrist': 0.95
}

frog_tail_range = {
    'scale_x': [1, 3],
    'scale_y': [0.5, 1.5],
    'scale_z': [0.8, 1.5],
    'sunken': [0.8, 1.2],
    'sunken_limit': [0.5, 0.7],
    'breast': [0, 0],
    'body_curve': [0.7, 0.8],
    'wrist': [0.9, 0.98]
}

for k, v in frog_tail_range.items():
    frog_tail_range[k] = np.array(v)


lizard_upper_head = {
    'scale_x': 0.8,
    'scale_y': 0.3,
    'scale_z': 0.4,
    'blunt_head': 0.3,
    'up_head_position': 0.4,
    'up_head_degree': 0.2,
    'offset_x': 0,
    'offset_y': 0,
    'offset_z': 0,
}
lizard_upper_head_range = {
    'scale_x': [0.5, 1],
    'scale_y': [0, 0],
    'scale_z': [0, 0],
    'blunt_head': [0, 0.8],
    'up_head_position': [0.2, 0.5],
    'up_head_degree': [0.1, 0.3]
}
for k, v in lizard_upper_head_range.items():
    lizard_upper_head_range[k] = np.array(v)

lizard_lower_head = {
    'scale_x': 0.8,
    'scale_y': 0.3,
    'scale_z': 0.15,
    'blunt_head': 0.3,
    'up_head_position': 0.4,
    'up_head_degree': 0.2,
    'offset_x': 0,
    'offset_y': 0,
    'offset_z': 0,
}
lizard_lower_head_range = {
    'scale_x': [0.5, 1],
    'scale_y': [0, 0],
    'scale_z': [0, 0],
    'blunt_head': [0, 0.8],
    'up_head_position': [0.2, 0.5],
    'up_head_degree': [0.1, 0.3]
}
for k, v in lizard_lower_head_range.items():
    lizard_lower_head_range[k] = np.array(v)

ReptileHeadBody.param_templates = {
    'lizard': {
        'head': lizard_upper_head, 
        'hrange': lizard_upper_head_range,
        'tail': lizard_tail, 
        'trange': lizard_tail_range,
    }, 
    'snake': {
        'head': lizard_upper_head, 
        'hrange': lizard_upper_head_range,
        'tail': snake_tail, 
        'trange': snake_tail_range
    }, 
    'dinosaur': {
        'head': lizard_upper_head, 
        'hrange': lizard_upper_head_range,
        'tail': dinosaur_tail, 
        'trange': dinosaur_tail_range
    }, 
    'frog': {
        'head': lizard_upper_head, 
        'hrange': lizard_upper_head_range,
        'tail': frog_tail, 
        'trange': frog_tail_range
    }
}

ReptileBody.param_templates = {
    'lizard': {
        'head': lizard_upper_head, 
        'hrange': lizard_upper_head_range,
        'tail': lizard_tail, 
        'trange': lizard_tail_range,
    }, 
    'snake': {
        'head': lizard_upper_head, 
        'hrange': lizard_upper_head_range,
        'tail': snake_tail, 
        'trange': snake_tail_range
    }, 
    'dinosaur': {
        'head': lizard_upper_head, 
        'hrange': lizard_upper_head_range,
        'tail': dinosaur_tail, 
        'trange': dinosaur_tail_range
    }, 
    'frog': {
        'head': lizard_upper_head, 
        'hrange': lizard_upper_head_range,
        'tail': frog_tail, 
        'trange': frog_tail_range
    }
}

ReptileLowerHead.param_templates = {
    'head': lizard_lower_head, 
    'range': lizard_lower_head_range
}

ReptileUpperHead.param_templates = {
    'head': lizard_upper_head, 
    'range': lizard_upper_head_range
}

LizardFrontLeg.param_templates = {
    # 'lizard': {'leg': {}, 'range': {}},
    'lizard': {
        'leg': {
            'scale_x': 1,
            'scale_y': 0.2,
            'scale_z': 0.2,
        }, 
        'range': {
            'scale_x': [0.5, 1],
            'scale_y': [0.15, 0.3],
            'scale_z': [0.05, 0.2],
        }
    }
}
LizardBackLeg.param_templates = {
    # 'lizard': {'leg': {}, 'range': {}},
    'lizard': {
        'leg': {
            'scale_x': 1.5,
            'scale_y': 0.2,
            'scale_z': 0.2,
        }, 
        'range': {
            'scale_x': [1.5, 3],
            'scale_y': [0.15, 0.3],
            'scale_z': [0.05, 0.2],
        }
    }
}
LizardToe.param_templates['lizard'] = {'toe': {}, 'range': {}}