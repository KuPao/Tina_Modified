from sympy.core import symbol
from ..common import *
from ..mesh.prim import *
from sympy import *
import math
# from sympy import *

# 圓柱圓心兩點 p0(x0, y0, z0), p1(x1, y1, z1)
# [(x0 - x)+(x1 - x0)*t]^2 + [(y0 - y)+(y1 - y0)*t]^2 + [(z0 - z)+(z1 - z0)*t]^2 = r^2

@ti.data_oriented
class PrimitiveFormulation:
    def __init__(self, PrimitiveMesh, mass):
        self.mass = mass
        numpy_vert = PrimitiveMesh.verts.to_numpy()
        shape = numpy_vert.shape
        append_ones = np.ones((shape[0],shape[1],1), float)
        numpy_vert=np.append(numpy_vert,append_ones, axis = 2)
        t = PrimitiveMesh.trans.to_numpy()
        t = np.transpose(t)
        numpy_vert = numpy_vert@t
        numpy_vert = numpy_vert.reshape((shape[0]*shape[1], 4))
        
        # np.set_printoptions(threshold=np.inf)
        # print(numpy_vert)

        max_value = np.amax(numpy_vert, axis=0)
        min_value = np.amin(numpy_vert, axis=0)
        # max_index = np.argmax(numpy_vert, axis=0)
        # min_index = np.argmin(numpy_vert, axis=0)

        # print(numpy_vert[139][1])
        # print(numpy_vert[43][1])

        v0 = numpy_vert[0]
        v1 = numpy_vert[4]
        d0 = numpy_vert[139]
        d1 = numpy_vert[43]

        mid = np.array([(max_value[0]+min_value[0])/2, (max_value[1]+min_value[1])/2, (max_value[2]+min_value[2])/2])
        self.midpoint = mid
        
        vec = v0 - v1
        vec = np.delete(vec, 3, 0)
        len = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
        self.length = len
        self.r = np.linalg.norm(d0-d1)/2
        vec = vec / len

        self.center_top = mid + vec * len / 2
        self.center_bot = mid - vec * len / 2
        # print(self.center_top)
        # print(self.center_bot)
        # print(midpoint)
        # print(r)
        # print(center_top)
        # print(center_bot)

        x = symbols('x')
        y = symbols('y')
        z = symbols('z')

        self.f = ((self.center_top[1]-y)*(self.center_bot[2]-z)-(self.center_top[2]-z)*(self.center_bot[1]-y))**2 +((self.center_top[2]-z)*(self.center_bot[0]-x)-(self.center_top[0]-x)*(self.center_bot[2]-z))**2 + ((self.center_top[0]-x)*(self.center_bot[1]-y)-(self.center_top[1]-y)*(self.center_bot[0]-x))**2 - self.r**2 * ((self.center_bot[0]-self.center_top[0])**2+(self.center_bot[1]-self.center_top[1])**2+(self.center_bot[2]-self.center_top[2])**2)
        
    def Calculate_Collision(self, surface):
        x, y, z = symbols('x,y,z')
        #f = ((self.center_top[1]-y)*(self.center_bot[2]-z)-(self.center_top[2]-z)*(self.center_bot[1]-y))**2 +((self.center_top[2]-z)*(self.center_bot[0]-x)-(self.center_top[0]-x)*(self.center_bot[2]-z))**2 + ((self.center_top[0]-x)*(self.center_bot[1]-y)-(self.center_top[1]-y)*(self.center_bot[0]-x))**2 - self.r**2 * ((self.center_bot[0]-self.center_top[0])**2+(self.center_bot[1]-self.center_top[1])**2+(self.center_bot[2]-self.center_top[2])**2)
        result = solve([self.f, surface], (x, y, z))
        return result

    def Surface_Point(self, surface, normal, collision_point):
        x, y, z, t = symbols('x,y,z,t')
        param = self.f.subs(x, collision_point[0]+normal[0]*t).subs(y, collision_point[1]+normal[1]*t).subs(z, collision_point[2]+normal[2]*t)
        # print(param)
        sol = solve([param], t)
        result = []
        direction = 0
        distance = 0
        if(abs(sol[0][0])<abs(sol[1][0])):
            distance = sol[0][0]
            
        else:
            distance = sol[1][0]
        result.append(collision_point[0]+normal[0]*distance)
        result.append(collision_point[1]+normal[1]*distance)
        result.append(collision_point[2]+normal[2]*distance)
        if distance < 0:
            direction = 1
        elif distance > 0:
            direction = -1
        distance = abs(distance)
        return result, direction, distance