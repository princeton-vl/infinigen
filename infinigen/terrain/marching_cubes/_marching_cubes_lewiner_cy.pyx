# Copyright: 2009-2022 the scikit-image team
# License: BSD-3-Clause
# adapted by Zeyu Ma to avoid precision loss for large grids to achieve local determinism,
# therefore to make seamless stiched mesh

#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

"""
This is an implementation of the marching cubes algorithm proposed in:

Efficient implementation of Marching Cubes' cases with topological guarantees.
Thomas Lewiner, Helio Lopes, Antonio Wilson Vieira and Geovan Tavares.
Journal of Graphics Tools 8(2): pp. 1-15 (december 2003)

This algorithm has the advantage that it provides topologically correct
results, and the algorithms implementation is relatively simple. Most
of the magic is in the lookup tables, which are provided as open source.

Originally implemented in C++ by Thomas Lewiner in 2002, ported to Cython
by Almar Klein in 2012. Adapted for scikit-image in 2016.

"""

# Cython specific imports
import numpy as np
cimport numpy as cnp
cnp.import_array()

# Enable low level memory management
from libc.stdlib cimport malloc, free

# Define tiny winy number
cdef cnp.float64_t FLT_EPSILON = np.spacing(1.0) #0.0000001

# Define abs function for doubles
cdef inline cnp.float64_t dabs(cnp.float64_t a): return a if a>=0 else -a
cdef inline int imin(int a, int b): return a if a<b else b

# todo: allow dynamic isovalue?
# todo: can we disable Cython from checking for zero division? Sometimes we know that it never happens!

def remove_degenerate_faces(vertices, faces, *arrays):

    vertices_map0 = np.arange(len(vertices), dtype=np.int32)
    vertices_map1 = vertices_map0.copy()
    faces_ok = np.ones(len(faces), dtype=np.int32)

    cdef cnp.float32_t [:, :] vertices_ = vertices
    cdef cnp.float32_t [:] v1, v2, v3
    cdef int [:, :]  faces_ = faces

    cdef int [:] vertices_map1_ = vertices_map1
    cdef int [:] faces_ok_ = faces_ok

    cdef int j, i1, i2, i3

    # Iterate over all faces. When we encounter a degenerate triangle,
    # we update the vertex map, i.e. we merge the corresponding vertices.
    for j in range(faces_.shape[0]):
        i1, i2, i3 = faces_[j][0], faces_[j][1], faces_[j][2]
        v1, v2, v3 = vertices_[i1], vertices_[i2], vertices_[i3]
        if v1[0] == v2[0] and v1[1] == v2[1] and v1[2] == v2[2]:
            vertices_map1_[i1] = vertices_map1_[i2] = imin(vertices_map1_[i1], vertices_map1_[i2])
            faces_ok_[j] = 0
        if v1[0] == v3[0] and v1[1] == v3[1] and v1[2] == v3[2]:
            vertices_map1_[i1] = vertices_map1_[i3] = imin(vertices_map1_[i1], vertices_map1_[i3])
            faces_ok_[j] = 0
        if v2[0] == v3[0] and v2[1] == v3[1] and v2[2] == v3[2]:
            vertices_map1_[i2] = vertices_map1_[i3] = imin(vertices_map1_[i2], vertices_map1_[i3])
            faces_ok_[j] = 0

    # Create mask and mapping to new vertex indices
    vertices_ok = vertices_map1 == vertices_map0
    vertices_map2 = np.cumsum(vertices_ok) - 1

    # Apply selection and mapping
    faces2 = vertices_map2[vertices_map1[faces[faces_ok>0]]]
    vertices2 = vertices[vertices_ok]
    arrays2 = [arr[vertices_ok] for arr in arrays]

    return (vertices2, faces2) + tuple(arrays2)


cdef class Cell:
    """ Class to keep track of some stuff during the whole cube marching
    procedure.

    This "struct" keeps track of the current cell location, and the values
    of corners of the cube. Gradients for the cube corners are calculated
    when needed.

    Additionally, it keeps track of the array of vertices, faces and normals.

    Notes on vertices
    -----------------
    The vertices are stored in a C-array that is increased in size with
    factors of two if needed. The same applies to the faces and normals.

    Notes on faces
    --------------
    To keep track of the vertices already defined, this class maintains
    two faceLayer arrays. faceLayer1 is of the current layer (z-value)
    and faceLayer2 is of the next. Both face layers have 4 elements per
    cell in that layer, 1 for each unique edge per cell (see
    get_index_in_facelayer). These are initialized as -1, and set to the
    index in the vertex array when a new vertex is created.
    In summary, this allows us to keep track of the already created
    vertices without keeping a very big array.

    Notes on normals
    ----------------
    The normal is simply defined as the gradient. Each time that a face is
    created, we also add the gradient of that vertex position to the
    normals array. The gradients are all calculated from the differences between
    the 8 corners of the current cube, but because the final value of a normal
    was contributed from multiple cells, the normals are quite accurate.

    """

    # Reference to LUTS object
    cdef LutProvider luts

    # Location of cube
    cdef int x
    cdef int y
    cdef int z

    # Stepsize
    cdef int step

    # Values of cube corners (isovalue subtracted)
    cdef cnp.float64_t v0
    cdef cnp.float64_t v1
    cdef cnp.float64_t v2
    cdef cnp.float64_t v3
    cdef cnp.float64_t v4
    cdef cnp.float64_t v5
    cdef cnp.float64_t v6
    cdef cnp.float64_t v7

    # Small arrays to store the above values in (allowing indexing)
    # and also the gradient at these points
    cdef cnp.float64_t *vv
    cdef cnp.float64_t *vg

    # Max value of the eight corners
    cdef cnp.float64_t vmax

    # Vertex position of center of cube (only calculated if needed)
    cdef cnp.float64_t v12_x_fractal
    cdef cnp.float64_t v12_y_fractal
    cdef cnp.float64_t v12_z_fractal
    # And corresponding gradient
    cdef cnp.float64_t v12_xg
    cdef cnp.float64_t v12_yg
    cdef cnp.float64_t v12_zg
    cdef int v12_calculated # a boolean

    # The index value, our magic 256 bit word
    cdef int index

    # Dimensions of the total volume
    cdef int nx
    cdef int ny
    cdef int nz

    # Arrays with face information
    cdef int *faceLayer # The current facelayer (reference-copy of one of the below)
    cdef int *faceLayer1 # The actual first face layer
    cdef int *faceLayer2 # The actual second face layer

    # Stuff to store the output vertices
    cdef cnp.float32_t *_vertices_fractal
    cdef cnp.int32_t *_vertices_integral
    cdef cnp.float32_t *_normals
    cdef cnp.float32_t *_values
    cdef int _vertexCount
    cdef int _vertexMaxCount

    # Stuff to store the output faces
    cdef int *_faces
    cdef int _faceCount
    cdef int _faceMaxCount


    def __init__(self, LutProvider luts, int nx, int ny, int nz):
        self.luts = luts
        self.nx, self.ny, self.nz = nx, ny, nz

        # Allocate face layers
        self.faceLayer1 = <int *>malloc(self.nx*self.ny*4 * sizeof(int))
        self.faceLayer2 = <int *>malloc(self.nx*self.ny*4 * sizeof(int))

        if (self.faceLayer1 is NULL or self.faceLayer2 is NULL or
            self.vv is NULL or self.vg is NULL or self._vertices_fractal is NULL or self._vertices_integral is NULL or
            self._normals is NULL or self._values is NULL or
            self._faces is NULL):
            raise MemoryError()

        cdef int i
        for i in range(self.nx*self.ny*4):
            self.faceLayer1[i] = -1
            self.faceLayer2[i] = -1
        self.faceLayer = self.faceLayer1


    def __cinit__(self):

        # Init tiny arrays for vertices and gradients at the vertices
        self.vv = <cnp.float64_t *>malloc(8 * sizeof(cnp.float64_t))
        self.vg = <cnp.float64_t *>malloc(8*3 * sizeof(cnp.float64_t))

        # Init face layers
        self.faceLayer1 = NULL
        self.faceLayer2 = NULL

        # Init vertices
        self._vertexCount = 0
        self._vertexMaxCount = 8
        self._vertices_fractal = <cnp.float32_t *>malloc(self._vertexMaxCount*3 * sizeof(cnp.float32_t))
        self._vertices_integral = <cnp.int32_t *>malloc(self._vertexMaxCount*3 * sizeof(cnp.int32_t))
        self._normals = <cnp.float32_t *>malloc(self._vertexMaxCount*3 * sizeof(cnp.float32_t))
        self._values = <cnp.float32_t *>malloc(self._vertexMaxCount * sizeof(cnp.float32_t))
        # Clear normals and values
        cdef int i, j
        if self._values is not NULL and self._normals is not NULL:
            for i in range(self._vertexMaxCount):
                self._values[i] = 0.0
                for j in range(3):
                    self._normals[i*3+j] = 0.0

        # Init faces
        self._faceCount = 0
        self._faceMaxCount = 8
        self._faces = <int *>malloc(self._faceMaxCount * sizeof(int))


    def __dealloc__(self):
        free(self.vv)
        free(self.vg)
        free(self.faceLayer1)
        free(self.faceLayer2)
        free(self._vertices_fractal)
        free(self._vertices_integral)
        free(self._normals)
        free(self._values)
        free(self._faces)


    cdef void _increase_size_vertices(self):
        """ Increase the size of the vertices array by a factor two.
        """
        # Allocate new array
        cdef int newMaxCount = self._vertexMaxCount * 2
        cdef cnp.float32_t *newVertices_fractal = <cnp.float32_t *>malloc(newMaxCount*3 * sizeof(cnp.float32_t))
        cdef cnp.int32_t *newVertices_integral = <cnp.int32_t *>malloc(newMaxCount*3 * sizeof(cnp.int32_t))
        cdef cnp.float32_t *newNormals = <cnp.float32_t *>malloc(newMaxCount*3 * sizeof(cnp.float32_t))
        cdef cnp.float32_t *newValues = <cnp.float32_t *>malloc(newMaxCount * sizeof(cnp.float32_t))
        if newVertices_fractal is NULL or newVertices_integral is NULL or newNormals is NULL or newValues is NULL:
            free(newVertices_integral)
            free(newVertices_fractal)
            free(newNormals)
            free(newValues)
            raise MemoryError()
        # Clear
        cdef int i, j
        for i in range(self._vertexCount, newMaxCount):
            newValues[i] = 0.0
            for j in range(3):
                newNormals[i*3+j] = 0.0
        # Copy
        for i in range(self._vertexCount):
            newValues[i] = self._values[i]
            for j in range(3):
                newVertices_fractal[i*3+j] = self._vertices_fractal[i*3+j]
                newVertices_integral[i*3+j] = self._vertices_integral[i*3+j]
                newNormals[i*3+j] = self._normals[i*3+j]
        # Apply
        free(self._vertices_fractal); self._vertices_fractal = newVertices_fractal
        free(self._vertices_integral); self._vertices_integral = newVertices_integral
        free(self._normals); self._normals = newNormals
        free(self._values); self._values = newValues
        self._vertexMaxCount = newMaxCount


    cdef void _increase_size_faces(self):
        """ Increase the size of the faces array by a factor two.
        """
        # Allocate new array
        cdef int newMaxCount = self._faceMaxCount * 2
        cdef int *newFaces = <int *>malloc(newMaxCount * sizeof(int))
        if newFaces is NULL:
            raise MemoryError()
        # Copy
        cdef int i
        for i in range(self._faceCount):
            newFaces[i] = self._faces[i]
        # Apply
        free(self._faces)
        self._faces = newFaces
        self._faceMaxCount = newMaxCount


    ## Adding results

    cdef int add_vertex(self, cnp.float32_t x, cnp.float32_t y,
                        cnp.float32_t z):
        """ Add a vertex to the result. Return index in vertex array.
        """
        # Check if array is large enough
        if self._vertexCount >= self._vertexMaxCount:
            self._increase_size_vertices()
        # Add vertex
        self._vertices_integral[self._vertexCount*3+0] = self.x
        self._vertices_integral[self._vertexCount*3+1] = self.y
        self._vertices_integral[self._vertexCount*3+2] = self.z
        self._vertices_fractal[self._vertexCount*3+0] = x
        self._vertices_fractal[self._vertexCount*3+1] = y
        self._vertices_fractal[self._vertexCount*3+2] = z
        self._vertexCount += 1
        return self._vertexCount -1


    cdef void add_gradient(self, int vertexIndex, cnp.float32_t gx,
                           cnp.float32_t gy, cnp.float32_t gz):
        """ Add a gradient value to the vertex corresponding to the given index.
        """
        self._normals[vertexIndex*3+0] += gx
        self._normals[vertexIndex*3+1] += gy
        self._normals[vertexIndex*3+2] += gz


    cdef void add_gradient_from_index(self, int vertexIndex, int i,
                                      cnp.float32_t strength):
        """ Add a gradient value to the vertex corresponding to the given index.
        vertexIndex is the index in the large array of vertices that is returned.
        i is the index of the array of vertices 0-7 for the current cell.
        """
        self.add_gradient(vertexIndex, self.vg[i*3+0] * strength,
                          self.vg[i*3+1] * strength, self.vg[i*3+2] * strength)


    cdef add_face(self, int index):
        """ Add a face to the result. Also updates the value.
        """
        # Check if array is large enough
        if self._faceCount >= self._faceMaxCount:
            self._increase_size_faces()
        # Add face
        self._faces[self._faceCount] = index
        self._faceCount += 1
        # Also update value
        if self.vmax > self._values[index]:
            self._values[index] = self.vmax


    ## Getting results

    def get_vertices(self):
        """ Get the final vertex array.
        """
        vertices_integral = np.empty((self._vertexCount,3), np.int32)
        vertices_fractal = np.empty((self._vertexCount,3), np.float32)
        cdef cnp.int32_t [:, :] vertices_integral_ = vertices_integral
        cdef cnp.float32_t [:, :] vertices_fractal_ = vertices_fractal
        cdef int i, j
        for i in range(self._vertexCount):
            for j in range(3):
                vertices_integral_[i, j] = self._vertices_integral[i*3+j]
                vertices_fractal_[i, j] = self._vertices_fractal[i*3+j]
        return vertices_integral, vertices_fractal

    def get_normals(self):
        """ Get the final normals array.
        The normals are normalized to unit length.
        """
        normals = np.empty((self._vertexCount,3), np.float32)
        cdef cnp.float32_t [:, :] normals_ = normals

        cdef int i, j
        cdef cnp.float64_t length, dtmp
        for i in range(self._vertexCount):
            length = 0.0
            for j in range(3):
                dtmp = self._normals[i*3+j] # Make it double before taking **2!
                length +=  dtmp*dtmp
            if length > 0.0:
                length = 1.0 / length**0.5
            for j in range(3):
                normals_[i,j] = self._normals[i*3+j] * length
        return normals

    def get_faces(self):
        faces = np.empty((self._faceCount,), np.int32)
        cdef int [:] faces_ = faces
        cdef int i
        for i in range(self._faceCount):
            faces_[i] = self._faces[i]
        return faces

    def get_values(self):
        values = np.empty((self._vertexCount,), np.float32)
        cdef cnp.float32_t [:] values_ = values
        cdef int i
        for i in range(self._vertexCount):
            values_[i] = self._values[i]
        return values


    ## Called from marching cube function

    cdef void new_z_value(self):
        """ This method should be called each time a new z layer is entered.
        We will swap the layers with face information and empty the second.
        """
        # Swap layers
        self.faceLayer1, self.faceLayer2 = self.faceLayer2, self.faceLayer1
        # Empty last
        cdef int i
        for i in range(self.nx*self.ny*4):
            self.faceLayer2[i] = -1


    cdef void set_cube(self,    cnp.float64_t isovalue, int x, int y, int z, int step,
                                cnp.float64_t v0, cnp.float64_t v1, cnp.float64_t v2, cnp.float64_t v3,
                                cnp.float64_t v4, cnp.float64_t v5, cnp.float64_t v6, cnp.float64_t v7):
        """ Set the cube to the new location.

        Set the values of the cube corners. The isovalue is subtracted
        from them, such that in further calculations the isovalue can be
        taken as zero.

        This method also calculated the magic 256 word to identify the
        cases (i.e. cell.index).
        """

        # Set location and step
        self.x = x
        self.y = y
        self.z = z
        self.step = step

        # Set values
        self.v0 = v0 - isovalue
        self.v1 = v1 - isovalue
        self.v2 = v2 - isovalue
        self.v3 = v3 - isovalue
        self.v4 = v4 - isovalue
        self.v5 = v5 - isovalue
        self.v6 = v6 - isovalue
        self.v7 = v7 - isovalue

        # Calculate index
        cdef int index = 0
        if self.v0 > 0.0:   index += 1
        if self.v1 > 0.0:   index += 2
        if self.v2 > 0.0:   index += 4
        if self.v3 > 0.0:   index += 8
        if self.v4 > 0.0:   index += 16
        if self.v5 > 0.0:   index += 32
        if self.v6 > 0.0:   index += 64
        if self.v7 > 0.0:   index += 128
        self.index = index

        # Reset c12
        self.v12_calculated = 0


    cdef void add_triangles(self, Lut lut, int lutIndex, int nt):
        """ Add triangles.

        The vertices for the triangles are specified in the given
        Lut at the specified index. There are nt triangles.

        The reason that nt should be given is because it is often known
        beforehand.

        """

        cdef int i, j
        cdef int vi

        self.prepare_for_adding_triangles()

        for i in range(nt):
            for j in range(3):
                # Get two sides for each element in this vertex
                vi = lut.get2(lutIndex, i*3+j)
                self._add_face_from_edge_index(vi)


    cdef void add_triangles2(self, Lut lut, int lutIndex, int lutIndex2, int nt):
        """ Same as add_triangles, except that now the geometry is in a LUT
        with 3 dimensions, and an extra index is provided.

        """
        cdef int i, j
        cdef int vi

        self.prepare_for_adding_triangles()

        for i in range(nt):
            for j in range(3):
                # Get two sides for each element in this vertex
                vi = lut.get3(lutIndex, lutIndex2, i*3+j)
                self._add_face_from_edge_index(vi)

    ## Used internally

    cdef void _add_face_from_edge_index(self, int vi):
        """ Add one face from an edge index. Only adds a face if the
        vertex already exists. Otherwise also adds a vertex and applies
        interpolation.
        """

        # typedefs
        cdef int indexInVertexArray, indexInFaceLayer
        cdef int dx1, dy1, dz1
        cdef int dx2, dy2, dz2
        cdef int index1, index2
        cdef cnp.float64_t tmpf1, tmpf2
        cdef cnp.float64_t fx, fy, fz, ff
        cdef cnp.float64_t stp = <cnp.float64_t>self.step

        # Get index in the face layer and corresponding vertex number
        indexInFaceLayer = self.get_index_in_facelayer(vi)
        indexInVertexArray = self.faceLayer[indexInFaceLayer]

        # If we have the center vertex, we have things pre-calculated,
        # otherwise we need to interpolate.
        # In both cases we distinguish between having this vertex already
        # or not.

        if vi == 12: # center vertex
            if self.v12_calculated == 0:
                self.calculate_center_vertex()
            if indexInVertexArray >= 0:
                # Vertex already calculated, only need to add face and gradient
                self.add_face(indexInVertexArray)
                self.add_gradient(indexInVertexArray, self.v12_xg, self.v12_yg, self.v12_zg)
            else:
                # Add precalculated center vertex position (is interpolated)
                indexInVertexArray = self.add_vertex( self.v12_x_fractal, self.v12_y_fractal, self.v12_z_fractal)
                # Update face layer
                self.faceLayer[indexInFaceLayer] = indexInVertexArray
                # Add face and gradient
                self.add_face(indexInVertexArray)
                self.add_gradient(indexInVertexArray, self.v12_xg, self.v12_yg, self.v12_zg)

        else:

            # Get relative edge indices for x, y and z
            dx1, dx2 = self.luts.EDGESRELX.get2(vi,0), self.luts.EDGESRELX.get2(vi,1)
            dy1, dy2 = self.luts.EDGESRELY.get2(vi,0), self.luts.EDGESRELY.get2(vi,1)
            dz1, dz2 = self.luts.EDGESRELZ.get2(vi,0), self.luts.EDGESRELZ.get2(vi,1)
            # Make two vertex indices
            index1 = dz1*4 + dy1*2 + dx1
            index2 = dz2*4 + dy2*2 + dx2
            # Define strength of both corners
            tmpf1 = 1.0 / (FLT_EPSILON + dabs(self.vv[index1]))
            tmpf2 = 1.0 / (FLT_EPSILON + dabs(self.vv[index2]))

            # print('indexInVertexArray', self.x, self.y, self.z, '-', vi, indexInVertexArray, indexInFaceLayer)

            if indexInVertexArray >= 0:
                # Vertex already calculated, only need to add face and gradient
                self.add_face(indexInVertexArray)
                self.add_gradient_from_index(indexInVertexArray, index1, tmpf1)
                self.add_gradient_from_index(indexInVertexArray, index2, tmpf2)

            else:
                # Interpolate by applying a kind of center-of-mass method
                fx, fy, fz, ff = 0.0, 0.0, 0.0, 0.0
                fx += <cnp.float64_t>dx1 * tmpf1;  fy += <cnp.float64_t>dy1 * tmpf1;  fz += <cnp.float64_t>dz1 * tmpf1;  ff += tmpf1
                fx += <cnp.float64_t>dx2 * tmpf2;  fy += <cnp.float64_t>dy2 * tmpf2;  fz += <cnp.float64_t>dz2 * tmpf2;  ff += tmpf2

                # Add vertex
                indexInVertexArray = self.add_vertex(stp*fx/ff, stp*fy/ff, stp*fz/ff)
                # Update face layer
                self.faceLayer[indexInFaceLayer] = indexInVertexArray
                # Add face and gradient
                self.add_face(indexInVertexArray)
                self.add_gradient_from_index(indexInVertexArray, index1, tmpf1)
                self.add_gradient_from_index(indexInVertexArray, index2, tmpf2)


#         # Create vertex non-interpolated
#         self.add_vertex( self.x + 0.5* dx1 + 0.5 * dx2,
#                         self.y + 0.5* dy1 + 0.5 * dy2,
#                         self.z + 0.5* dz1 + 0.5 * dz2 )



    cdef int get_index_in_facelayer(self, int vi):
        """
        Get the index of a vertex position, given the edge on which it lies.
        We keep a list of faces so we can reuse vertices. This improves
        speed because we need less interpolation, and the result is more
        compact and can be visualized better because normals can be
        interpolated.

        For each cell, we store 4 vertex indices; all other edges can be
        represented as the edge of another cell.  The fourth is the center vertex.

        This method returns -1 if no vertex has been defined yet.


              vertices              edges                edge-indices per cell
        *         7 ________ 6           _____6__             ________
        *         /|       /|         7/|       /|          /|       /|
        *       /  |     /  |        /  |     /5 |        /  |     /  |
        *   4 /_______ /    |      /__4____ /    10     /_______ /    |
        *    |     |  |5    |     |    11  |     |     |     |  |     |
        *    |    3|__|_____|2    |     |__|__2__|     |     |__|_____|
        *    |    /   |    /      8   3/   9    /      2    /   |    /
        *    |  /     |  /        |  /     |  /1       |  1     |  /
        *    |/_______|/          |/___0___|/          |/___0___|/
        *   0          1
        */
        """

        # Init indices, both are corrected below
        cdef int i = self.nx * self.y + self.x  # Index of cube to get vertex at
        cdef int j = 0 # Vertex number for that cell

        cdef int *faceLayer

        # Select either upper or lower half
        if vi < 8:
            #  8 horizontal edges
            if vi < 4:
                faceLayer = self.faceLayer1
            else:
                vi -= 4
                faceLayer = self.faceLayer2

            # Calculate actual index based on edge
            #if vi == 0: pass  # no step
            if vi == 1:  # step in x
                i += self.step
                j = 1
            elif vi == 2:  # step in y
                i += self.nx * self.step
            elif vi == 3:  # no step
                j = 1

        elif vi < 12:
            # 4 vertical edges
            faceLayer = self.faceLayer1
            j = 2

            #if vi == 8: pass # no step
            if vi == 9:   # step in x
                i += self.step
            elif vi == 10:   # step in x and y
                i += self.nx * self.step + self.step
            elif vi == 11:  # step in y
                i += self.nx * self.step

        else:
            # center vertex
            faceLayer = self.faceLayer1
            j = 3

        # Store facelayer and return index
        self.faceLayer = faceLayer # Dirty way of returning a value
        return 4*i + j


    cdef void prepare_for_adding_triangles(self):
        """ Calculates some things to help adding the triangles:
        array with corner values, max corner value, gradient at each corner.
        """

        cdef int i

        # Copy values in array so we can index them. Note the misalignment
        # because the numbering does not correspond with bitwise OR of xyz.
        self.vv[0] = self.v0
        self.vv[1] = self.v1
        self.vv[2] = self.v3#
        self.vv[3] = self.v2#
        self.vv[4] = self.v4
        self.vv[5] = self.v5
        self.vv[6] = self.v7#
        self.vv[7] = self.v6#

        # Calculate max
        cdef cnp.float64_t vmin, vmax
        vmin, vmax = 0.0, 0.0
        for i in range(8):
            if self.vv[i] > vmax:
                vmax = self.vv[i]
            if self.vv[i] < vmin:
                vmin = self.vv[i]
        self.vmax = vmax-vmin

        # Calculate gradients
        # Derivatives, selected to always point in same direction.
        # Note that many corners have the same components as other points,
        # by interpolating  and averaging the normals this is solved.
        # todo: we can potentially reuse these similar to how we store vertex indices in face layers
        self.vg[0*3+0], self.vg[0*3+1], self.vg[0*3+2] = self.v0-self.v1, self.v0-self.v3, self.v0-self.v4
        self.vg[1*3+0], self.vg[1*3+1], self.vg[1*3+2] = self.v0-self.v1, self.v1-self.v2, self.v1-self.v5
        self.vg[2*3+0], self.vg[2*3+1], self.vg[2*3+2] = self.v3-self.v2, self.v1-self.v2, self.v2-self.v6
        self.vg[3*3+0], self.vg[3*3+1], self.vg[3*3+2] = self.v3-self.v2, self.v0-self.v3, self.v3-self.v7
        self.vg[4*3+0], self.vg[4*3+1], self.vg[4*3+2] = self.v4-self.v5, self.v4-self.v7, self.v0-self.v4
        self.vg[5*3+0], self.vg[5*3+1], self.vg[5*3+2] = self.v4-self.v5, self.v5-self.v6, self.v1-self.v5
        self.vg[6*3+0], self.vg[6*3+1], self.vg[6*3+2] = self.v7-self.v6, self.v5-self.v6, self.v2-self.v6
        self.vg[7*3+0], self.vg[7*3+1], self.vg[7*3+2] = self.v7-self.v6, self.v4-self.v7, self.v3-self.v7


    cdef void calculate_center_vertex(self):
        """ Calculate interpolated center vertex and its gradient.
        """
        cdef cnp.float64_t v0, v1, v2, v3, v4, v5, v6, v7
        cdef cnp.float64_t fx, fy, fz, ff
        fx, fy, fz, ff = 0.0, 0.0, 0.0, 0.0

        # Define "strength" of each corner of the cube that we need
        v0 = 1.0 / (FLT_EPSILON + dabs(self.v0))
        v1 = 1.0 / (FLT_EPSILON + dabs(self.v1))
        v2 = 1.0 / (FLT_EPSILON + dabs(self.v2))
        v3 = 1.0 / (FLT_EPSILON + dabs(self.v3))
        v4 = 1.0 / (FLT_EPSILON + dabs(self.v4))
        v5 = 1.0 / (FLT_EPSILON + dabs(self.v5))
        v6 = 1.0 / (FLT_EPSILON + dabs(self.v6))
        v7 = 1.0 / (FLT_EPSILON + dabs(self.v7))

        # Apply a kind of center-of-mass method
        fx += 0.0*v0;  fy += 0.0*v0;  fz += 0.0*v0;  ff += v0
        fx += 1.0*v1;  fy += 0.0*v1;  fz += 0.0*v1;  ff += v1
        fx += 1.0*v2;  fy += 1.0*v2;  fz += 0.0*v2;  ff += v2
        fx += 0.0*v3;  fy += 1.0*v3;  fz += 0.0*v3;  ff += v3
        fx += 0.0*v4;  fy += 0.0*v4;  fz += 1.0*v4;  ff += v4
        fx += 1.0*v5;  fy += 0.0*v5;  fz += 1.0*v5;  ff += v5
        fx += 1.0*v6;  fy += 1.0*v6;  fz += 1.0*v6;  ff += v6
        fx += 0.0*v7;  fy += 1.0*v7;  fz += 1.0*v7;  ff += v7

        # Store
        cdef cnp.float64_t stp = <cnp.float64_t>self.step
        self.v12_x_fractal = stp * fx / ff
        self.v12_y_fractal = stp * fy / ff
        self.v12_z_fractal = + stp * fz / ff

        # Also pre-calculate gradient of center
        # note that prepare_for_adding_triangles() must have been called for
        # the gradient data to exist.
        self.v12_xg = ( v0*self.vg[0*3+0] + v1*self.vg[1*3+0] + v2*self.vg[2*3+0] + v3*self.vg[3*3+0] +
                        v4*self.vg[4*3+0] + v5*self.vg[5*3+0] + v6*self.vg[6*3+0] + v7*self.vg[7*3+0] )
        self.v12_yg = ( v0*self.vg[0*3+1] + v1*self.vg[1*3+1] + v2*self.vg[2*3+1] + v3*self.vg[3*3+1] +
                        v4*self.vg[4*3+1] + v5*self.vg[5*3+1] + v6*self.vg[6*3+1] + v7*self.vg[7*3+1] )
        self.v12_zg = ( v0*self.vg[0*3+2] + v1*self.vg[1*3+2] + v2*self.vg[2*3+2] + v3*self.vg[3*3+2] +
                        v4*self.vg[4*3+2] + v5*self.vg[5*3+2] + v6*self.vg[6*3+2] + v7*self.vg[7*3+2] )

        # Set flag that this stuff is calculated
        self.v12_calculated = 1



cdef class Lut:
    """ Representation of a lookup table.
    The tables are initially defined as numpy arrays. On initialization,
    this class converts them to a C array for fast access.
    This class defines functions to look up values using 1, 2 or 3 indices.
    """

    cdef signed char* VALUES
    cdef int L0 # Length
    cdef int L1 # size of tuple
    cdef int L2 # size of tuple in tuple (if any)

    def __init__(self, array):

        # Get the shape of the LUT
        self.L1 = 1
        self.L2 = 1
        #
        self.L0 = array.shape[0]
        if array.ndim > 1:
            self.L1 = array.shape[1]
        if array.ndim > 2:
            self.L2 = array.shape[2]

        # Copy the contents
        array = array.ravel()
        cdef int n, N
        N = self.L0 * self.L1 * self.L2
        self.VALUES = <signed char *> malloc(N * sizeof(signed char))
        if self.VALUES is NULL:
            raise MemoryError()
        for n in range(N):
            self.VALUES[n] = array[n]

    def __cinit__(self):
        self.VALUES = NULL

    def __dealloc__(self):
        if self.VALUES is not NULL:
            free(self.VALUES)

    cdef int get1(self, int i0):
        return self.VALUES[i0]

    cdef int get2(self, int i0, int i1):
        return self.VALUES[i0*self.L1 + i1]

    cdef int get3(self, int i0, int i1, int i2):
        return self.VALUES[i0*self.L1*self.L2 + i1*self.L2 + i2]



cdef class LutProvider:
    """ Class that provides a common interface to the many lookup tables
    used by the algorithm.
    All the lists of lut names are autogenerated to prevent human error.
    """

    cdef Lut EDGESRELX # Edges relative X
    cdef Lut EDGESRELY
    cdef Lut EDGESRELZ

    cdef Lut CASESCLASSIC
    cdef Lut CASES

    cdef Lut TILING1
    cdef Lut TILING2
    cdef Lut TILING3_1
    cdef Lut TILING3_2
    cdef Lut TILING4_1
    cdef Lut TILING4_2
    cdef Lut TILING5
    cdef Lut TILING6_1_1
    cdef Lut TILING6_1_2
    cdef Lut TILING6_2
    cdef Lut TILING7_1
    cdef Lut TILING7_2
    cdef Lut TILING7_3
    cdef Lut TILING7_4_1
    cdef Lut TILING7_4_2
    cdef Lut TILING8
    cdef Lut TILING9
    cdef Lut TILING10_1_1
    cdef Lut TILING10_1_1_
    cdef Lut TILING10_1_2
    cdef Lut TILING10_2
    cdef Lut TILING10_2_
    cdef Lut TILING11
    cdef Lut TILING12_1_1
    cdef Lut TILING12_1_1_
    cdef Lut TILING12_1_2
    cdef Lut TILING12_2
    cdef Lut TILING12_2_
    cdef Lut TILING13_1
    cdef Lut TILING13_1_
    cdef Lut TILING13_2
    cdef Lut TILING13_2_
    cdef Lut TILING13_3
    cdef Lut TILING13_3_
    cdef Lut TILING13_4
    cdef Lut TILING13_5_1
    cdef Lut TILING13_5_2
    cdef Lut TILING14

    cdef Lut TEST3
    cdef Lut TEST4
    cdef Lut TEST6
    cdef Lut TEST7
    cdef Lut TEST10
    cdef Lut TEST12
    cdef Lut TEST13

    cdef Lut SUBCONFIG13


    def __init__(self, EDGESRELX, EDGESRELY, EDGESRELZ, CASESCLASSIC, CASES,

            TILING1, TILING2, TILING3_1, TILING3_2, TILING4_1, TILING4_2,
            TILING5, TILING6_1_1, TILING6_1_2, TILING6_2, TILING7_1, TILING7_2,
            TILING7_3, TILING7_4_1, TILING7_4_2, TILING8, TILING9,
            TILING10_1_1, TILING10_1_1_, TILING10_1_2, TILING10_2, TILING10_2_,
            TILING11, TILING12_1_1, TILING12_1_1_, TILING12_1_2, TILING12_2,
            TILING12_2_, TILING13_1, TILING13_1_, TILING13_2, TILING13_2_,
            TILING13_3, TILING13_3_, TILING13_4, TILING13_5_1, TILING13_5_2,
            TILING14,

            TEST3, TEST4, TEST6, TEST7, TEST10, TEST12, TEST13,
            SUBCONFIG13,
            ):

        self.EDGESRELX = Lut(EDGESRELX)
        self.EDGESRELY = Lut(EDGESRELY)
        self.EDGESRELZ = Lut(EDGESRELZ)

        self.CASESCLASSIC = Lut(CASESCLASSIC)
        self.CASES = Lut(CASES)

        self.TILING1 = Lut(TILING1)
        self.TILING2 = Lut(TILING2)
        self.TILING3_1 = Lut(TILING3_1)
        self.TILING3_2 = Lut(TILING3_2)
        self.TILING4_1 = Lut(TILING4_1)
        self.TILING4_2 = Lut(TILING4_2)
        self.TILING5 = Lut(TILING5)
        self.TILING6_1_1 = Lut(TILING6_1_1)
        self.TILING6_1_2 = Lut(TILING6_1_2)
        self.TILING6_2 = Lut(TILING6_2)
        self.TILING7_1 = Lut(TILING7_1)
        self.TILING7_2 = Lut(TILING7_2)
        self.TILING7_3 = Lut(TILING7_3)
        self.TILING7_4_1 = Lut(TILING7_4_1)
        self.TILING7_4_2 = Lut(TILING7_4_2)
        self.TILING8 = Lut(TILING8)
        self.TILING9 = Lut(TILING9)
        self.TILING10_1_1 = Lut(TILING10_1_1)
        self.TILING10_1_1_ = Lut(TILING10_1_1_)
        self.TILING10_1_2 = Lut(TILING10_1_2)
        self.TILING10_2 = Lut(TILING10_2)
        self.TILING10_2_ = Lut(TILING10_2_)
        self.TILING11 = Lut(TILING11)
        self.TILING12_1_1 = Lut(TILING12_1_1)
        self.TILING12_1_1_ = Lut(TILING12_1_1_)
        self.TILING12_1_2 = Lut(TILING12_1_2)
        self.TILING12_2 = Lut(TILING12_2)
        self.TILING12_2_ = Lut(TILING12_2_)
        self.TILING13_1 = Lut(TILING13_1)
        self.TILING13_1_ = Lut(TILING13_1_)
        self.TILING13_2 = Lut(TILING13_2)
        self.TILING13_2_ = Lut(TILING13_2_)
        self.TILING13_3 = Lut(TILING13_3)
        self.TILING13_3_ = Lut(TILING13_3_)
        self.TILING13_4 = Lut(TILING13_4)
        self.TILING13_5_1 = Lut(TILING13_5_1)
        self.TILING13_5_2 = Lut(TILING13_5_2)
        self.TILING14 = Lut(TILING14)

        self.TEST3 = Lut(TEST3)
        self.TEST4 = Lut(TEST4)
        self.TEST6 = Lut(TEST6)
        self.TEST7 = Lut(TEST7)
        self.TEST10 = Lut(TEST10)
        self.TEST12 = Lut(TEST12)
        self.TEST13 = Lut(TEST13)

        self.SUBCONFIG13 = Lut(SUBCONFIG13)

def marching_cubes(cnp.float32_t[:, :, :] im not None, cnp.float64_t isovalue,
                   LutProvider luts, int st=1, int classic=0,
                   cnp.ndarray[cnp.npy_bool, ndim=3, cast=True] mask=None):
    """ marching_cubes(im, cnp.float64_t isovalue, LutProvider luts, int st=1, int classic=0)
    Main entry to apply marching cubes.

    Masked version of marching cubes. This function will check a
    masking array (same size as im) to decide if the algorithm must be
    computed for a given voxel. This adds a small overhead that
    rapidly gets compensated by the fewer computed cubes
    Returns (vertices, faces, normals, values)
    """
    # Get dimemsnions
    cdef int Nx, Ny, Nz
    Nx, Ny, Nz = im.shape[2], im.shape[1], im.shape[0]

    # Create cell to use throughout
    cdef Cell cell = Cell(luts, Nx, Ny, Nz)

    # Typedef variables
    cdef int x, y, z, x_st, y_st, z_st
    cdef int nt
    cdef int case, config
    cdef bint no_mask = mask is None
    # Unfortunately specifying a step in range() significantly degrades
    # performance. Therefore we use a while loop.
    # we have:  max_x = Nx_bound + st + st - 1
    #       ->  Nx_bound = max_allowable_x + 1 - 2 * st
    #       ->  Nx_bound = Nx - 2 * st
    assert st > 0
    cdef int Nx_bound, Ny_bound, Nz_bound
    Nx_bound, Ny_bound, Nz_bound = Nx - 2 * st, Ny - 2 * st, Nz - 2 * st  # precalculated index range

    z = -st
    while z < Nz_bound:
        z += st
        z_st = z + st

        cell.new_z_value()  # Indicate that we enter a new layer
        y = -st
        while y < Ny_bound:
            y += st
            y_st = y + st

            x = -st
            while x < Nx_bound:
                x += st
                x_st = x + st
                if no_mask or mask[z_st, y_st, x_st]:
                    # Initialize cell
                    cell.set_cube(isovalue, x, y, z, st,
                        im[z   ,y, x], im[z   ,y, x_st], im[z   ,y_st, x_st], im[z   ,y_st, x],
                        im[z_st,y, x], im[z_st,y, x_st], im[z_st,y_st, x_st], im[z_st,y_st, x] )

                    # Do classic!
                    if classic:
                        # Determine number of vertices
                        nt = 0
                        while luts.CASESCLASSIC.get2(cell.index, 3*nt) != -1:
                            nt += 1
                        # Add triangles
                        if nt > 0:
                            cell.add_triangles(luts.CASESCLASSIC, cell.index, nt)
                    else:
                        # Get case, if non-nul, enter the big switch
                        case = luts.CASES.get2(cell.index, 0)
                        if case > 0:
                            config = luts.CASES.get2(cell.index, 1)
                            the_big_switch(luts, cell, case, config)

    # Done
    return *cell.get_vertices(), cell.get_faces(), cell.get_normals(), cell.get_values()



cdef void the_big_switch(LutProvider luts, Cell cell, int case, int config):
    """ The big switch (i.e. if-statement) that I meticulously ported from
    the source code provided by Lewiner et. al.

    Together with all the look-up tables, this is where the magic is ...
    """

    cdef int subconfig = 0

    # Sinatures for tests
    #test_face(cell, luts.TESTX.get1(config)):
    #test_internal(cell, luts, case, config, subconfig, luts.TESTX.get1(config)):
    #cell.add_triangles(luts.TILINGX, config, N)

    if case == 1:
        cell.add_triangles(luts.TILING1, config, 1)

    elif case == 2:
        cell.add_triangles(luts.TILING2, config, 2)

    elif case == 3:
        if test_face(cell, luts.TEST3.get1(config)):
            cell.add_triangles(luts.TILING3_2, config, 4)
        else:
            cell.add_triangles(luts.TILING3_1, config, 2)

    elif case == 4 :
        if test_internal(cell, luts, case, config, subconfig, luts.TEST4.get1(config)):
            cell.add_triangles(luts.TILING4_1, config, 2)
        else:
            cell.add_triangles(luts.TILING4_2, config, 6)

    elif case == 5 :
        cell.add_triangles(luts.TILING5, config, 3)

    elif case == 6 :
        if test_face(cell, luts.TEST6.get2(config,0)):
            cell.add_triangles(luts.TILING6_2, config, 5)
        else:
            if test_internal(cell, luts, case, config, subconfig, luts.TEST6.get2(config,1)):
                cell.add_triangles(luts.TILING6_1_1, config, 3)
            else:
                #cell.calculate_center_vertex() # v12 needed
                cell.add_triangles(luts.TILING6_1_2, config, 9)

    elif case == 7 :
        # Get subconfig
        if test_face(cell, luts.TEST7.get2(config,0)): subconfig += 1
        if test_face(cell, luts.TEST7.get2(config,1)): subconfig += 2
        if test_face(cell, luts.TEST7.get2(config,2)): subconfig += 4
        # Behavior depends on subconfig
        if subconfig == 0: cell.add_triangles(luts.TILING7_1, config, 3)
        elif subconfig == 1: cell.add_triangles2(luts.TILING7_2, config, 0, 5)
        elif subconfig == 2: cell.add_triangles2(luts.TILING7_2, config, 1, 5)
        elif subconfig == 3:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING7_3, config, 0, 9)
        elif subconfig == 4: cell.add_triangles2(luts.TILING7_2, config, 2, 5)
        elif subconfig == 5:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING7_3, config, 1, 9)
        elif subconfig == 6:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING7_3, config, 2, 9)
        elif subconfig == 7:
            if test_internal(cell, luts, case, config, subconfig, luts.TEST7.get2(config,3)):
                cell.add_triangles(luts.TILING7_4_2, config, 9)
            else:
                cell.add_triangles(luts.TILING7_4_1, config, 5)

    elif case == 8 :
        cell.add_triangles(luts.TILING8, config, 2)

    elif case == 9 :
        cell.add_triangles(luts.TILING9, config, 4)

    elif case == 10 :
        if test_face(cell, luts.TEST10.get2(config,0)):
            if test_face(cell, luts.TEST10.get2(config,1)):
                cell.add_triangles(luts.TILING10_1_1_, config, 4)
            else:
                #cell.calculate_center_vertex() # v12 needed
                cell.add_triangles(luts.TILING10_2, config, 8)
        else:
            if test_face(cell, luts.TEST10.get2(config,1)):
                #cell.calculate_center_vertex() # v12 needed
                cell.add_triangles(luts.TILING10_2_, config, 8)
            else:
                if test_internal(cell, luts, case, config, subconfig, luts.TEST10.get2(config,2)):
                    cell.add_triangles(luts.TILING10_1_1, config, 4)
                else:
                    cell.add_triangles(luts.TILING10_1_2, config, 8)

    elif case == 11 :
        cell.add_triangles(luts.TILING11, config, 4)

    elif case == 12 :
        if test_face(cell, luts.TEST12.get2(config,0)):
            if test_face(cell, luts.TEST12.get2(config,1)):
                cell.add_triangles(luts.TILING12_1_1_, config, 4)
            else:
                #cell.calculate_center_vertex() # v12 needed
                cell.add_triangles(luts.TILING12_2, config, 8)
        else:
            if test_face(cell, luts.TEST12.get2(config,1)):
                #cell.calculate_center_vertex() # v12 needed
                cell.add_triangles(luts.TILING12_2_, config, 8)
            else:
                if test_internal(cell, luts, case, config, subconfig, luts.TEST12.get2(config,2)):
                    cell.add_triangles(luts.TILING12_1_1, config, 4)
                else:
                    cell.add_triangles(luts.TILING12_1_2, config, 8)

    elif case == 13 :
        # Calculate subconfig
        if test_face(cell, luts.TEST13.get2(config,0)): subconfig += 1
        if test_face(cell, luts.TEST13.get2(config,1)): subconfig += 2
        if test_face(cell, luts.TEST13.get2(config,2)): subconfig += 4
        if test_face(cell, luts.TEST13.get2(config,3)): subconfig += 8
        if test_face(cell, luts.TEST13.get2(config,4)): subconfig += 16
        if test_face(cell, luts.TEST13.get2(config,5)): subconfig += 32

        # Map via LUT
        subconfig = luts.SUBCONFIG13.get1(subconfig)

        # Behavior depends on subconfig
        if subconfig==0:    cell.add_triangles(luts.TILING13_1, config, 4)
        elif subconfig==1:  cell.add_triangles2(luts.TILING13_2, config, 0, 6)
        elif subconfig==2:  cell.add_triangles2(luts.TILING13_2, config, 1, 6)
        elif subconfig==3:  cell.add_triangles2(luts.TILING13_2, config, 2, 6)
        elif subconfig==4:  cell.add_triangles2(luts.TILING13_2, config, 3, 6)
        elif subconfig==5:  cell.add_triangles2(luts.TILING13_2, config, 4, 6)
        elif subconfig==6:  cell.add_triangles2(luts.TILING13_2, config, 5, 6)
        #
        elif subconfig==7:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_3, config, 0, 10)
        elif subconfig==8:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_3, config, 1, 10)
        elif subconfig==9:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_3, config, 2, 10)
        elif subconfig==10:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_3, config, 3, 10)
        elif subconfig==11:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_3, config, 4, 10)
        elif subconfig==12:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_3, config, 5, 10)
        elif subconfig==13:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_3, config, 6, 10)
        elif subconfig==14:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_3, config, 7, 10)
        elif subconfig==15:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_3, config, 8, 10)
        elif subconfig==16:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_3, config, 9, 10)
        elif subconfig==17:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_3, config, 10, 10)
        elif subconfig==18:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_3, config, 11, 10)
        #
        elif subconfig==19:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_4, config, 0, 12)
        elif subconfig==20:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_4, config, 1, 12)
        elif subconfig==21:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_4, config, 2, 12)
        elif subconfig==22:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_4, config, 3, 12)
        #
        elif subconfig==23:
            subconfig = 0 # Note: the original source code sets the subconfig, without apparent reason
            if test_internal(cell, luts, case, config, subconfig, luts.TEST13.get2(config,6)):
                cell.add_triangles2(luts.TILING13_5_1, config, 0, 6)
            else:
                cell.add_triangles2(luts.TILING13_5_2, config, 0, 10)
        elif subconfig==24:
            subconfig = 1
            if test_internal(cell, luts, case, config, subconfig, luts.TEST13.get2(config,6)):
                cell.add_triangles2(luts.TILING13_5_1, config, 1, 6)
            else:
                cell.add_triangles2(luts.TILING13_5_2, config, 1, 10)
        elif subconfig==25:
            subconfig = 2 ;
            if test_internal(cell, luts, case, config, subconfig, luts.TEST13.get2(config,6)):
                cell.add_triangles2(luts.TILING13_5_1, config, 2, 6)
            else:
                cell.add_triangles2(luts.TILING13_5_2, config, 2, 10)
        elif subconfig==26:
            subconfig = 3 ;
            if test_internal(cell, luts, case, config, subconfig, luts.TEST13.get2(config,6)):
                cell.add_triangles2(luts.TILING13_5_1, config, 3, 6)
            else:
                cell.add_triangles2(luts.TILING13_5_2, config, 3, 10)
        #
        elif subconfig==27:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_3_, config, 0, 10)
        elif subconfig==28:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_3_, config, 1, 10)
        elif subconfig==29:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_3_, config, 2, 10)
        elif subconfig==30:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_3_, config, 3, 10)
        elif subconfig==31:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_3_, config, 4, 10)
        elif subconfig==32:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_3_, config, 5, 10)
        elif subconfig==33:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_3_, config,6, 10)
        elif subconfig==34:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_3_, config, 7, 10)
        elif subconfig==35:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_3_, config, 8, 10)
        elif subconfig==36:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_3_, config, 9, 10)
        elif subconfig==37:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_3_, config, 10, 10)
        elif subconfig==38:
            #cell.calculate_center_vertex() # v12 needed
            cell.add_triangles2(luts.TILING13_3_, config, 11, 10)
        #
        elif subconfig==39:
            cell.add_triangles2(luts.TILING13_2_, config, 0, 6)
        elif subconfig==40:
            cell.add_triangles2(luts.TILING13_2_, config, 1, 6)
        elif subconfig==41:
            cell.add_triangles2(luts.TILING13_2_, config, 2, 6)
        elif subconfig==42:
            cell.add_triangles2(luts.TILING13_2_, config, 3, 6)
        elif subconfig==43:
            cell.add_triangles2(luts.TILING13_2_, config, 4, 6)
        elif subconfig==44:
            cell.add_triangles2(luts.TILING13_2_, config, 5, 6)
        #
        elif subconfig==45:
            cell.add_triangles(luts.TILING13_1_, config, 4)
        #
        else:
            print("Marching Cubes: Impossible case 13?" )

    elif case == 14 :
        cell.add_triangles(luts.TILING14, config, 4)


cdef int test_face(Cell cell, int face):
    """ Return True of the face contains part of the surface.
    """

    # Get face absolute value
    cdef int absFace = face
    if face < 0:
        absFace *= -1

    # Get values of corners A B C D
    cdef cnp.float64_t A=0, B=0, C=0, D=0
    if absFace == 1:
        A, B, C, D = cell.v0, cell.v4, cell.v5, cell.v1
    elif absFace == 2:
        A, B, C, D = cell.v1, cell.v5, cell.v6, cell.v2
    elif absFace == 3:
        A, B, C, D = cell.v2, cell.v6, cell.v7, cell.v3
    elif absFace == 4:
        A, B, C, D = cell.v3, cell.v7, cell.v4, cell.v0
    elif absFace == 5:
        A, B, C, D = cell.v0, cell.v3, cell.v2, cell.v1
    elif absFace == 6:
        A, B, C, D = cell.v4, cell.v7, cell.v6, cell.v5

    # Return sign
    cdef cnp.float64_t AC_BD = A*C - B*D
    if AC_BD > - FLT_EPSILON and AC_BD < FLT_EPSILON:
        return face >= 0
    else:
        return face * A * AC_BD >= 0;  # face and A invert signs


cdef int test_internal(Cell cell, LutProvider luts, int case, int config, int subconfig, int s):
    """ Return True of the face contains part of the surface.
    """

    # Typedefs
    cdef cnp.float64_t t, At, Bt, Ct, Dt, a, b
    cdef int test = 0
    cdef int edge = -1 # reference edge of the triangulation


    # Calculate At Bt Ct Dt a b
    # Select case 4, 10,  7, 12, 13
    At, Bt, Ct, Dt = 0.0, 0.0, 0.0, 0.0

    if case==4 or case==10:
        a = ( cell.v4 - cell.v0 ) * ( cell.v6 - cell.v2 ) - ( cell.v7 - cell.v3 ) * ( cell.v5 - cell.v1 )
        b =  cell.v2 * ( cell.v4 - cell.v0 ) + cell.v0 * ( cell.v6 - cell.v2 ) - cell.v1 * ( cell.v7 - cell.v3 ) - cell.v3 * ( cell.v5 - cell.v1 )
        t = - b / (2*a + FLT_EPSILON)
        if t<0 or t>1:  return s>0 ;

        At = cell.v0 + ( cell.v4 - cell.v0 ) * t
        Bt = cell.v3 + ( cell.v7 - cell.v3 ) * t
        Ct = cell.v2 + ( cell.v6 - cell.v2 ) * t
        Dt = cell.v1 + ( cell.v5 - cell.v1 ) * t

    elif case==6 or case==7 or case==12 or case==13:
        # Define edge
        if case == 6:  edge = luts.TEST6.get2(config, 2)
        elif case == 7: edge = luts.TEST7.get2(config, 4)
        elif case == 12: edge = luts.TEST12.get2(config, 3)
        elif case == 13: edge = luts.TILING13_5_1.get3(config, subconfig, 0)

        if edge==0:
            t  = cell.v0 / ( cell.v0 - cell.v1 + FLT_EPSILON )
            At = 0
            Bt = cell.v3 + ( cell.v2 - cell.v3 ) * t
            Ct = cell.v7 + ( cell.v6 - cell.v7 ) * t
            Dt = cell.v4 + ( cell.v5 - cell.v4 ) * t
        elif edge==1:
            t  = cell.v1 / ( cell.v1 - cell.v2 + FLT_EPSILON )
            At = 0
            Bt = cell.v0 + ( cell.v3 - cell.v0 ) * t
            Ct = cell.v4 + ( cell.v7 - cell.v4 ) * t
            Dt = cell.v5 + ( cell.v6 - cell.v5 ) * t
        elif edge==2:
            t  = cell.v2 / ( cell.v2 - cell.v3 + FLT_EPSILON )
            At = 0
            Bt = cell.v1 + ( cell.v0 - cell.v1 ) * t
            Ct = cell.v5 + ( cell.v4 - cell.v5 ) * t
            Dt = cell.v6 + ( cell.v7 - cell.v6 ) * t
        elif edge==3:
            t  = cell.v3 / ( cell.v3 - cell.v0 + FLT_EPSILON )
            At = 0
            Bt = cell.v2 + ( cell.v1 - cell.v2 ) * t
            Ct = cell.v6 + ( cell.v5 - cell.v6 ) * t
            Dt = cell.v7 + ( cell.v4 - cell.v7 ) * t
        elif edge==4:
            t  = cell.v4 / ( cell.v4 - cell.v5 + FLT_EPSILON )
            At = 0
            Bt = cell.v7 + ( cell.v6 - cell.v7 ) * t
            Ct = cell.v3 + ( cell.v2 - cell.v3 ) * t
            Dt = cell.v0 + ( cell.v1 - cell.v0 ) * t
        elif edge==5:
            t  = cell.v5 / ( cell.v5 - cell.v6 + FLT_EPSILON )
            At = 0
            Bt = cell.v4 + ( cell.v7 - cell.v4 ) * t
            Ct = cell.v0 + ( cell.v3 - cell.v0 ) * t
            Dt = cell.v1 + ( cell.v2 - cell.v1 ) * t
        elif edge==6:
            t  = cell.v6 / ( cell.v6 - cell.v7 + FLT_EPSILON )
            At = 0
            Bt = cell.v5 + ( cell.v4 - cell.v5 ) * t
            Ct = cell.v1 + ( cell.v0 - cell.v1 ) * t
            Dt = cell.v2 + ( cell.v3 - cell.v2 ) * t
        elif edge==7:
            t  = cell.v7 / ( cell.v7 - cell.v4 + FLT_EPSILON )
            At = 0
            Bt = cell.v6 + ( cell.v5 - cell.v6 ) * t
            Ct = cell.v2 + ( cell.v1 - cell.v2 ) * t
            Dt = cell.v3 + ( cell.v0 - cell.v3 ) * t
        elif edge==8:
            t  = cell.v0 / ( cell.v0 - cell.v4 + FLT_EPSILON )
            At = 0
            Bt = cell.v3 + ( cell.v7 - cell.v3 ) * t
            Ct = cell.v2 + ( cell.v6 - cell.v2 ) * t
            Dt = cell.v1 + ( cell.v5 - cell.v1 ) * t
        elif edge==9:
            t  = cell.v1 / ( cell.v1 - cell.v5 + FLT_EPSILON )
            At = 0
            Bt = cell.v0 + ( cell.v4 - cell.v0 ) * t
            Ct = cell.v3 + ( cell.v7 - cell.v3 ) * t
            Dt = cell.v2 + ( cell.v6 - cell.v2 ) * t
        elif edge==10:
            t  = cell.v2 / ( cell.v2 - cell.v6 + FLT_EPSILON )
            At = 0
            Bt = cell.v1 + ( cell.v5 - cell.v1 ) * t
            Ct = cell.v0 + ( cell.v4 - cell.v0 ) * t
            Dt = cell.v3 + ( cell.v7 - cell.v3 ) * t
        elif edge==11:
            t  = cell.v3 / ( cell.v3 - cell.v7 + FLT_EPSILON )
            At = 0
            Bt = cell.v2 + ( cell.v6 - cell.v2 ) * t
            Ct = cell.v1 + ( cell.v5 - cell.v1 ) * t
            Dt = cell.v0 + ( cell.v4 - cell.v0 ) * t
        else:
            print( "Invalid edge %i." % edge )
    else:
        print( "Invalid ambiguous case %i." % case )

    # Process results
    if At >= 0: test += 1
    if Bt >= 0: test += 2
    if Ct >= 0: test += 4
    if Dt >= 0: test += 8

    # Determine what to return
    if test==0: return s>0
    elif test==1: return s>0
    elif test==2: return s>0
    elif test==3: return s>0
    elif test==4: return s>0
    elif test==5:
        if At * Ct - Bt * Dt <  FLT_EPSILON: return s>0
    elif test==6: return s>0
    elif test==7: return s<0
    elif test==8: return s>0
    elif test==9: return s>0
    elif test==10:
        if At * Ct - Bt * Dt >= FLT_EPSILON: return s>0
    elif test==11: return s<0
    elif test==12: return s>0
    elif test==13: return s<0
    elif test==14: return s<0
    elif test==15: return s<0
    else: return s<0
