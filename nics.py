import matplotlib.pyplot as plt
import moleculetools as mt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class SurfaceStructure(mt.Structure):
    def update_relevant_coords(self):
        """
        Updates a list of important atoms (since not all of them are wanted to
        define a surface)
        """
        self.relevant_coords = self.coords[self.atom_index_list]

    def choose_atoms(self, atom_list=None):
        """
        Define a list of important atoms to use for surface fitting
        """
        if atom_list is None:
            atom_list = list(range(1, len(self.atoms) + 1))
        self.atom_index_list = [index - 1 for index in atom_list]
        self.update_relevant_coords()

    def update_geometry(self):
        """
        Overwrite update_geometry() to take into account only the chosen coords
        Center and main axis will be defined based on them
        """
        self.center = mt.find_center(self.relevant_coords)
        self.update_relevant_coords()
        self.main_axis = mt.best_fitted_plane(self.relevant_coords)
        self.update_relevant_coords()

    def make_surface(self, radius=None, density=50, distance=0, functions=None, method='linear'):
        """
        Generates a fitted surface based on the chosen atoms
        """
        if radius == None:
            self.find_radius()
            radius = self.radius

        if functions == None:
            functions = [
                ("x", lambda xy:xy[:,0]),
                ("y", lambda xy:xy[:,1]),
                ("x^2", lambda xy:np.square(xy[:,0])),
                ("y^2", lambda xy:np.square(xy[:,1])),
                ("xy", lambda xy:np.multiply(xy[:,0], xy[:,1])),
                ("sin(x)", lambda xy:np.sin(xy[:,0])),
                ("sin(y)", lambda xy:np.sin(xy[:,1])),
                ("sqrt(x^2 + y^2)", lambda xy:np.sqrt(np.square(xy[:,0]) + np.square(xy[:,1]))),
                ("sqrt(x^2 + y^2)^2", lambda xy:np.sqrt(np.square(xy[:,0]) + np.square(xy[:,1]))**2),
                ("sin(x^2 + y^2)", lambda xy:np.sin(xy[:,0]**2 + xy[:,1]**2)),
                #("1/(x^2 + y^2)", lambda xy:1/(xy[:,0]**2 + xy[:,1]**2))
            ]

        def build_predictors(x, y, functions):
            xy = np.column_stack([x, y])
            X_list = []
            for f in functions:
                X_list.append(f[1](xy))
            return np.column_stack(X_list)

        def print_results(model, y_true, y_pred, functions):
            print("\nLinear regression results:")
            for i, f in enumerate(functions):
                print("{:>20}: {:>10.7f}".format(f[0], model.coef_[i]))
            print("\nR2 score: {:8f}".format(r2_score(y_true, y_pred)))

        def mid_x_validation(model, functions):
            midpoint_list = []
            for bond in self.bonds:
                coord1 = self.coords[bond['atoms'][0]]
                coord2 = self.coords[bond['atoms'][1]]
                midpoint_list.append((coord1 + coord2) / 2)
            self.midpoints = np.vstack(midpoint_list)
            X = build_predictors(self.midpoints[:,0], self.midpoints[:,1], functions)
            y = self.midpoints[:,2]
            r2 = r2_score(y, model.predict(X))
            print("\nR2 score of atom midpoints: {:8f}".format(r2))

        if method == 'linear':
            lr = LinearRegression(normalize=True)
            X = build_predictors(self.relevant_coords[:,0],
                                 self.relevant_coords[:,1], functions)
            y = self.relevant_coords[:,2]
            lr.fit(X, y)
            grid = make_grid(radius=radius, density=density)
            pred_grid = build_predictors(grid[:,0], grid[:,1], functions)
            preds = lr.predict(pred_grid) + distance
            self.surface = np.column_stack((grid, preds))
            print_results(lr, lr.predict(X), y, functions)
            mid_x_validation(lr, functions)


    def save(self, draw_axis=True, draw_surface=True, numbering=True,
             filename=None, interactive=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.coords[:,0],
                   self.coords[:,1],
                   self.coords[:,2], marker='o', s=30, color="black")
        ax.scatter(self.midpoints[:,0],
                   self.midpoints[:,1],
                   self.midpoints[:,2], marker='o', s=30, color="green")
        if numbering:
            for i in range(self.coords.shape[0]):
                ax.text(self.coords[i,0],
                        self.coords[i,1],
                        self.coords[i,2],
                        str(i + 1))
        if draw_axis:
            n_points = 21
            axis_points = np.reshape(np.linspace(-5, 5, n_points), (n_points, 1))
            points = np.matmul(axis_points, np.reshape(self.main_axis, (1, 3))) + self.center
            ax.scatter(points[:,0], points[:,1], points[:,2], marker='.')
        try:
            ax.scatter(self.surface[:,0],
                       self.surface[:,1],
                       self.surface[:,2],
                       marker='.', s=5, alpha=0.5, c=self.surface[:,2], cmap="viridis")
        except:
            pass

        ax.set_xlabel(r'x ($\AA$)')
        ax.set_ylabel(r'y ($\AA$)')
        ax.set_zlabel(r'z ($\AA$)')

        try:
            set_axes_radius(ax, self.center, self.surface[:,0].max())
        except:
            pass

        ax_limit = np.ceil(self.radius*2)/2

        ax.set_xlim3d(-ax_limit, ax_limit)
        ax.set_ylim3d(-ax_limit, ax_limit)
        ax.set_zlim3d(-ax_limit, ax_limit)

        if filename == None:
            fig.savefig("{}_surface.png".format(self.name), transparent=True)
        else:
            fig.savefig("{}{}.png".format(self.name, filename), transparent=True)
        if interactive:
            plt.show()
        plt.close()

    def write_gjf(self, surface=None):
        link0 = "# nmr=giao b3lyp/6-31G* nosymm geom=connectivity guess=huckel"
        with open("{}.gjf".format(self.name), "w") as gjf:
            gjf.write("{}\n\ninput\n\n0 1\n".format(link0))
            for i, atom in enumerate(self.atoms):
                gjf.write("{:<2}{:>14.8f}{:>14.8f}{:>14.8f}\n".format(atom,
                                                                self.coords[i,0],
                                                                self.coords[i,1],
                                                                self.coords[i,2]))
            if surface is not None:
                for i in range(surface.shape[0]):
                    gjf.write("{:<2}{:>14.8f}{:>14.8f}{:>14.8f}\n".format("Bq",
                                                                    surface[i,0],
                                                                    surface[i,1],
                                                                    surface[i,2]))
                gjf.write("\n")
                for i in range(len(self.atoms) + surface.shape[0]):
                    gjf.write("{}\n".format(i + 1))
            #gjf.write("\n")
            #for i in range(len(self.atoms) + len(self.surface)):
            #    gjf.write("{}\n".format(i + 1))

class ReadSurfaceStructure(mt.Structure):
    def split_coords(self):
        surface_coords = []
        atom_coords = []
        for i, atom in enumerate(self.atoms):
            if atom == "Bq":
                surface_coords.append(self.coords[i,:])
            else:
                atom_coords.append(self.coords[i,:])
        self.surface_coords = np.array(surface_coords)
        self.atom_coords = np.array(atom_coords)
        self.center = np.array([0, 0, 0])

    def load_nics(self, log):
        with open(log, "r") as open_log:
            log_lines = open_log.readlines()
        isodata = -np.array([float(line.split()[4])
                             for line in log_lines
                             if "Bq   Isotropic" in line])
        #for i, item in enumerate(isodata):
        #    if abs(item) > 2000:
        #        isodata[i] = item/abs(item)
        self.isodata = isodata

    def find_radius(self):
        distances = np.linalg.norm(self.atom_coords - self.center, axis=1)
        self.radius = distances.max()

    def save_3d(self, numbering=False, interactive=False):
        """
        Saves the plot of the NICS values over the atoms of the molecule as
        a 3D graph
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.atom_coords[:,0],
                   self.atom_coords[:,1],
                   self.atom_coords[:,2], marker='o', s=30, color="black")
        if numbering:
            for i in range(self.atom_coords.shape[0]):
                ax.text(self.atom_coords[i,0],
                        self.atom_coords[i,1],
                        self.atom_coords[i,2],
                        str(i + 1))
        try:
            s = ax.scatter(self.surface_coords[:,0],
                       self.surface_coords[:,1],
                       self.surface_coords[:,2],
                       marker='.', s=15, alpha=1,
                       c=self.isodata,
                       vmin=-np.amax(abs(self.isodata)),
                       vmax=np.amax(abs(self.isodata)),
                       cmap="seismic")
        except:
            pass

        ax.set_xlabel(r'x ($\AA$)')
        ax.set_ylabel(r'y ($\AA$)')
        ax.set_zlabel(r'z ($\AA$)')

        ax_limit = np.ceil(self.radius*2)/2

        ax.set_xlim3d(-ax_limit, ax_limit)
        ax.set_ylim3d(-ax_limit, ax_limit)
        ax.set_zlim3d(-ax_limit, ax_limit)

        fig.colorbar(s, orientation='vertical', fraction=0.03)
        fig.savefig("{}-3d.png".format(self.name), transparent=True)
        if interactive:
            plt.show()
        plt.close()

    def save_2d(self, numbering=False):
        """
        Saves the plot of the NICS values over the atoms of the molecule as
        a 2D graph where the molecule is flattened along its main axis (z)
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if numbering:
            for i in range(self.atom_coords.shape[0]):
                ax.text(self.atom_coords[i,0],
                        self.atom_coords[i,1],
                        str(i + 1))
        #try:
        #    s = ax.tricontourf(self.surface_coords[:,0],
        #               self.surface_coords[:,1],
        #               marker='.', s=15, alpha=1,
        #               c=self.isodata,
        #               vmin=-np.amax(abs(self.isodata)),
        #               vmax=np.amax(abs(self.isodata)),
        #               cmap="seismic")

        vmin = -np.amax(abs(self.isodata))
        vmax = np.amax(abs(self.isodata))

        levels = np.linspace(vmin, vmax,
                             np.sqrt(self.surface_coords.shape[0]))

        tricontourf_ = ax.tricontourf(self.surface_coords[:,0],
                       self.surface_coords[:,1],
                       self.isodata, levels=levels, vmin=vmin, vmax=vmax,
                       cmap='seismic')

        ax.scatter(self.atom_coords[:,0],
                   self.atom_coords[:,1], marker='o', s=30, color="black")

        ax.margins(0,0)
        ax.axis("scaled")
        ax.set_xlim(np.amin(self.surface_coords[:,0]),
                    np.amax(self.surface_coords[:,0]))
        ax.set_ylim(np.amin(self.surface_coords[:,1]),
                    np.amax(self.surface_coords[:,1]))

        ax.set_xlabel(r'x ($\AA$)')
        ax.set_ylabel(r'y ($\AA$)')

        cbar = fig.colorbar(tricontourf_,
                            orientation='vertical',
                            fraction=0.03,
                            extend='both', label="NICS (ppm)",
                            ticks=[np.ceil(vmin), 0, np.floor(vmax)])
        plt.tight_layout()
        fig.savefig("{}-2d.png".format(self.name), transparent=True)
        plt.close()


def parse_atom_list(atom_list_str=None):
    """
    Interprets custom atom list strings and returns true lists of integers
    """
    if atom_list_str == None:
        return None
    atom_ranges = atom_list_str.split(",")
    atom_list = []
    for group in atom_ranges:
        try:
            limits = [int(x) for x in list(group.split("-"))]
            atom_list += list(range(limits[0] - 1, limits[1]))
        except:
            atom_list.append(int(group))
    print(atom_list)
    return(atom_list)

def read_log(log):
    """
    Reads elements and coordinates from a log file
    """
    with open(log, 'r') as open_log:
        log_lines = open_log.readlines()[2:]
    atom_list = []
    xyz_list = []
    getting_coords = False
    for i, line in enumerate(log_lines):
        if "Charge = " in line:
            getting_coords = True
        elif getting_coords:
            if len(line.strip()) == 0:
                break
            atom_list.append(line.split()[0])
            xyz_list.append([float(coord) for coord in line.split()[1:]])
    return (atom_list, np.asarray(xyz_list))

def make_grid(radius=5, density=50):
    """
    Helper function to create grids given a 'radius' (half-side of a square)
    and a density (points per side), returns a 2D numpy array
    """
    spacing = np.linspace(-radius, radius, density)
    X2 = np.meshgrid(spacing, spacing)
    grid_shape = X2[0].shape
    return np.reshape(X2, (2, -1)).T

# Execution
import click

@click.group()
def main():
    pass

# STEP 1: needs an xyz file, a list of atoms, and surface options
@main.command()
@click.argument('xyz_file')
@click.option('--atom-list', '-a', type=str, default=None,
              help='List of atoms to fit the surface to, e.g. "1-5,7,10-15"')
@click.option('--radius', '-r', type=float, default=None,
              help="Radius of the grid, default is automatic")
@click.option('--density', '-d', default=50, help="Density of the grid",
              type=int, show_default=True)
def surface(xyz_file, atom_list, radius, density):
    """
    Generates a fitted surface given an xyz file, a list of atoms and optional
    surface parameters
    """
    print("Build surface mode")
    print("Density = {}".format(density))

    system_id = xyz_file[:-4]
    els, coords = mt.read_xyz(xyz_file)
    system = SurfaceStructure(els, coords, name=system_id)
    system.choose_atoms(parse_atom_list(atom_list))
    system.update_geometry()
    system.translate_to_center()
    system.update_geometry()
    system.rotate_to_z()
    system.update_geometry()
    system.find_bonds()
    system.rotate_along_z(0)
    system.update_geometry()
    system.make_surface(radius=radius, density=density, distance=1)
    system.save(interactive=True)
    system.write_gjf(system.surface)

# STEP 2: needs a log file, and plotting options
@main.command()
@click.argument('log_file')
def plot(log_file):
    """
    Generates plots of the results given a log file
    """
    print("Plot results mode")

    system_id = log_file[:-4]
    els, coords = read_log(log_file)
    solved = ReadSurfaceStructure(els, coords, name=system_id)
    solved.split_coords()
    solved.load_nics(log_file)
    solved.find_radius()
    solved.save_3d(interactive=True)
    solved.save_2d()

@main.command()
@click.argument('xyz_file')
def test(xyz_file):
    system_id = xyz_file[:-4]
    els, coords = mt.read_xyz(xyz_file)
    system = SurfaceStructure(els, coords, name=system_id)
    system.choose_atoms()
    system.update_geometry()
    system.translate_to_center()
    system.update_geometry()
    system.rotate_to_z()
    system.update_geometry()
    system.find_bonds()

    angle = 6.28/100
    for num in range(1, 101):
        system.rotate_along_z(angle)
        system.update_geometry()
        system.make_surface()
        system.save(filename="-{}{}".format("0"*(4 - len(str(num))), num))
    #system.rotate_along_z(0.8)
    #system.update_geometry()
    #system.make_surface()
    #system.save()

if __name__ == "__main__":
    main()
