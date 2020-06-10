import numpy as np

element_dict = {}

def unit_vector(vector):
    """
    Returns the unit vector of vector
    """
    return vector/np.linalg.norm(vector)

def angle_between(u, v):
    """
    Returns the angle in radians between vectors u and v
    """
    u_u = unit_vector(u)
    v_u = unit_vector(v)
    return np.arccos(np.clip(np.dot(u_u, v_u), -1, 1))

def cos_between(u, v):
    """
    Returns the cosine between vectors u and v
    """
    u_u = unit_vector(u)
    v_u = unit_vector(v)
    return np.clip(np.dot(u_u, v_u), -1, 1)

def find_normal_from_points(a, b, c):
    """
    Finds the normal unit vector to a plane defined by a set of three points
    """
    dir = np.cross((b - a), (c - a))
    normal = unit_vector(dir)
    return normal

def find_normal_from_vectors(u, v):
    """
    Finds the normal unit vector to a plane defined by a set of two vectors
    """
    dir = np.cross(u, v)
    normal = unit_vector(dir)
    return normal

def rotation_matrix(angle, direction, point=None):
    sin = np.sin(angle)
    cos = np.cos(angle)
    direction = unit_vector(direction[:3])
    R = np.diag([cos, cos, cos])
    R += np.outer(direction, direction) * (1.0 - cos)
    direction *= sin
    R += np.array([[ 0.0,         -direction[2],  direction[1]],
                   [ direction[2], 0.0,          -direction[0]],
                   [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M

def apply_4x4_matrix_to_3D_set(matrix, coords):
    if coords.shape[1] == 3:
        coords = np.hstack((coords, np.ones((coords.shape[0], 1))))
    else:
        pass
    return matrix.dot(coords.T).T[:,:3]

#def calc_rot_matrix(current_axis, desired_axis, points=None):
#    rot_axis = np.cross(current_axis, desired_axis)
#    w = unit_vector(rot_axis)
#    wx, wy, wz = w
#    as_w = np.array([[  0, -wz,  wy],
#                     [ wz,   0, -wx],
#                     [-wy,  wx,   0]])
#    ang = angle_between(current_axis, desired_axis)
#    cos = cos_between(current_axis, desired_axis)
#    sin = np.sqrt(1 - cos**2)
#    R = np.identity(3) + as_w*sin + np.square(as_w)*(1 - cos)
#    return R

def find_center(coords):
    x = coords[:,0].mean()
    y = coords[:,1].mean()
    z = coords[:,2].mean()
    return np.array([x, y, z])

def best_fitted_plane(coords):
    centroid = find_center(coords)
    rel_coords = coords - centroid
    cov = np.cov(rel_coords.T)
    evals, evecs = np.linalg.eig(cov)
    return evecs[:,np.argmin(evals)]

#def find_axis(coords):
#    indices = list(range(coords.shape[0]))
#    combs = combinations(indices, 3)
#    unique_combs = list(set(combs))
#    norm_list = []
#    for comb in unique_combs:
#        a = coords[comb[0],:]
#        b = coords[comb[1],:]
#        c = coords[comb[2],:]
#        norm = find_normal_from_points(a, b, c)
#        norm_list.append(norm)
#    norm_array = np.array(norm_list)
#    norm_mean = np.mean(norm_array, axis=0)
#    return unit_vector(norm_mean)

class Structure:
    def __init__(self, atoms, coords, **kwargs):
        self.name = kwargs.get('name', "system")
        self.atoms = atoms
        self.coords = coords
        self.natoms = len(self.atoms)

    def find_center(self):
        x = self.coords[:,0].mean()
        y = self.coords[:,1].mean()
        z = self.coords[:,2].mean()
        self.center = np.array([x, y, z])

    def translate_to_center(self):
        self.update_geometry()
        trans_matrix = np.matmul(np.ones((self.natoms, 1)),
                                 np.reshape(self.center, (1, 3)))
        self.coords = self.coords - trans_matrix
        self.update_geometry()

    def find_radius(self):
        distances = np.linalg.norm(self.coords - self.center, axis=1)
        self.radius = distances.max()

    def find_bonds(self, min_l=1.3, max_l=2.1):
        MIN_BOND_LENGTH = min_l
        MAX_BOND_LENGTH = max_l
        self.bonds = []
        for i in range(len(self.atoms)):
            for j in range(i + 1, len(self.atoms)):
                if i != j:
                    d = np.linalg.norm(self.coords[i,:] - self.coords[j,:])
                    if MIN_BOND_LENGTH <= d <= MAX_BOND_LENGTH and not (self.atoms[i] == 'H' and self.atoms[j] == 'H'):
                        self.bonds.append({'atoms': (i, j),
                                           'elements': (self.atoms[i], self.atoms[j]),
                                           'distance': d})

    def find_axis(self):
        try:
            self.main_axis = best_fitted_plane(self.coords)
        except np.linalg.LinAlgError:
            self.main_axis = np.array([0.0, 0.0, 1.0])

    def update_geometry(self):
        self.find_center()
        self.find_axis()

    def rotate_to_z(self):
        self.update_geometry()
        z = np.array([0.0, 0.0, 1.0])
        angle = angle_between(self.main_axis, z)
        if angle > 1:
            direction = find_normal_from_vectors(self.main_axis, z)
            M = rotation_matrix(angle, direction, point=self.center)
            self.coords = apply_4x4_matrix_to_3D_set(M, self.coords)
        else:
            pass
        self.update_geometry()

    def rotate_along_z(self, angle):
        self.update_geometry()
        z = np.array([0, 0, 1])
        M = rotation_matrix(angle, z, point=self.center)
        self.coords = apply_4x4_matrix_to_3D_set(M, self.coords)
        self.update_geometry()


def read_xyz(xyz):
    with open(xyz, 'r') as open_xyz:
        xyz_lines = open_xyz.readlines()[2:]
    atom_list = []
    xyz_list = []
    for line in xyz_lines:
        try:
            atom_list.append(line.split()[0])
            xyz_list.append([float(coord) for coord in line.split()[1:]])
        except IndexError:
            pass
    return (atom_list, np.asarray(xyz_list))

    #def find_axis(self, atoms=None, print_info=False):
    #    if atoms is None:
    #        atoms = list(range(self.natoms))
    #    combs = combinations(atoms, 3)
    #    unique_combs = list(set(combs))
    #    norm_list = []
    #    for comb in unique_combs:
    #        a = self.coords[comb[0],:]
    #        b = self.coords[comb[1],:]
    #        c = self.coords[comb[2],:]
    #        norm = find_normal_from_points(a, b, c)
    #        norm_list.append(abs(unit_vector(norm)))
    #    norm_array = np.array(norm_list)
    #    norm_mean = np.mean(norm_array, axis=0)
    #    self.main_axis = unit_vector(norm_mean)
