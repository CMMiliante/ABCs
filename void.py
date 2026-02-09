# Copyright 2026 Caio Miranda Miliante

import copy
import itertools
import more_itertools
import time
import os
import numpy as np
import pandas as pd
import pymatgen.core
from pymatgen.core import Structure
import networkx as nx
from pymatgen.io.vasp import VolumetricData


class Void:
    """
    Class for the calculation of the void spaces present in a crystal cell.
    """

    def __init__(self, structure: pymatgen.core.Structure, output_dir_path: str, **kwargs):
        """
        :param structure: pymatgen.core.Structure parameter containing the information of the cell to be parsed
        :param output_dir_path: directory path for files to be exported to
        :param kwargs: keyword arguments for alternative names for the file to be exported and void calculation
         parameters
        """
        # Assign and declare instance variables
        self.structure, self.output_dir_path = structure, output_dir_path
        self.time_void, self.perc_calc_time = 0.0, 0.0
        self.graph, self.perc_dist, self.max_perc_disp = nx.Graph(), None, None
        self.structure_filename = kwargs.get('structure_filename', '{}.vasp'.format(structure.formula.replace(' ', '')))
        self.void_filename = kwargs.get('void_filename',
                                        'VOID_LOCPOT.vasp')  # File name to facilitate visualization in vesta
        self.maxperc_filename = kwargs.get('void_filename', 'MAX_PERCPATH.vasp')
        self._max_vol = kwargs.get('max_vol', 1E4)  # in Angstrom

        # Set the maximum spacing between points in the grid for void calculation and initializes the grid
        self._grid_spacing = kwargs.get('grid_spacing', .1)
        self.grid = Grid(self.structure.lattice, self._grid_spacing)

        # Initializes the nodes for the percolation path calculation
        self._nodes_points = copy.copy(self.grid.points)
        self._nodes_array = np.reshape(self._nodes_points, self.grid.natoms_per_vec, order='F')

        # Get the paths for the max percalation path and percolation path report files
        self._perc_filepath = os.path.join(self.output_dir_path, self.maxperc_filename)
        self._perc_report_path = os.path.join(self.output_dir_path, 'PercolationPath-Report.txt')

        # Export the structure of the material being simulated in a poscar format
        self._structure_filepath = os.path.join(self.output_dir_path, self.structure_filename)
        if not os.path.isfile(self._structure_filepath):
            self.structure.to_file(self._structure_filepath, fmt='poscar')

        # Check if a void file exists with a grid_spacing as strict as the one set, if not the void is recalculated.
        self._void_filepath = os.path.join(self.output_dir_path, self.void_filename)
        _void_calc = False
        if os.path.isfile(self._void_filepath):
            try:
                self.void_voldata = VolumetricData.parse_file(self._void_filepath)
            except IndexError:
                raise IndexError('Error parsing volumetric data file for voids - {}'.format(self.void_filename))

            max_spacing = max([length / self.void_voldata[1]['total'].shape[n] for n, length in
                               enumerate(self.void_voldata[0].structure.lattice.abc)])

            # Check for the void spacing utilized in the previous calculation
            if max_spacing > self._grid_spacing:
                print('\tVOID file available, but with larger spacing. Voids recalculated.')
                _void_calc = True

            else:
                print('\tVOID file with appropriate spacing available. Voids not recalculated.')

                # Incorporate the previous calculated distances into the grid points
                void_voldata_dists = self.void_voldata[1]['total'].reshape(-1, order='F')
                for (n_point, point) in enumerate(self.grid.points):
                    point.atom_dist = void_voldata_dists[n_point]

        else:
            print('\tVOID file not available. Voids to be calculated.')
            _void_calc = True

        # Perform the void calculation
        if _void_calc:
            # Void calculation
            self._calculate_void()

            # Void data export
            point_dist_matrix = np.array(list(zip(*self.void))[0]).reshape(self.grid.coordinates.shape[:-1],
                                                                           order='F')
            self.void_voldata = VolumetricData(structure=self.structure, data={'total': point_dist_matrix})
            self.void_voldata.write_file(self._void_filepath)

    @property
    def void(self):
        """
        Information for the void grid points.

        :return:
            list: list with the distance from a grid point to an atom in the structure and the grid point coordinates
        in the cell for all grid points considered.
        """
        return [(atom.atom_dist, atom.coordinate) for n_atom, atom in enumerate(self.grid.points)]

    def void_info(self, dists_eval: np.array = np.arange(.5, 7, .5)):
        """
        Information regarding the percentage of voids in the crystal cell.

        :param dists_eval: array with the distances to be evaluated for void percentage
        :return:
            dict: data for percentage of void in the cell at different evaluated distances
        """
        # Create the void_results dictionary containing the information for the time taken to calculate the void space
        void_results = {'time_void': self.time_void}

        # Iterate over the requested void distances (in Å) and check the percentage of the cell that is at least the
        # requested distance away from an atom in the structure
        for dist in dists_eval:
            void_results['void_perc_{:1.1f}'.format(dist).replace('.', 'p')] = sum(
                [1 if atom.atom_dist >= dist else 0 for atom in self.grid.points]) / len(self.grid.points)

        return void_results

    def perc_info(self):
        """
        Data stored regarding the percolation path calculation.

        :return:
            dict: Data contained in the percolation report
        """

        try:
            perc_report = pd.read_csv(self._perc_report_path)
        except FileNotFoundError:
            raise FileNotFoundError('Percolation report file could not be found. Please run the calculation for '
                                    'the percolation path first.')

        return perc_report.iloc[0].to_dict()

    @property
    def grid_spacing(self):
        """
        Information for the grid spacing considered in the calculation.

        :return:
            float: grid spacing.
        """
        return self._grid_spacing

    @grid_spacing.setter
    def grid_spacing(self, value: float):
        """
        Setter for the grid_spacing property.

        :param value: grid spacing to be set. A ValueError will be raised if a negative value is tried to be set.

        """
        if value < 0:
            raise ValueError("Grid spacing cannot be negative.")
        self._grid_spacing = value

    def percolation(self, perc_disp_thresholds: np.array = np.linspace(1, 0, 11),
                    perc_path_recalc: bool = False, **kwargs):
        """
        Evaluates if the percolation path has already been calculated, and if not calls for the calculation of the
        percolation paths.

        :param perc_disp_thresholds: array containing the displacements to be considered during the percolation path
        calculation
        :param perc_path_recalc: boolean to force the recalculation of the percolation path
        :param kwargs: keyword arguments to be passed on to the percolation path calculation
        """

        print('\t-- Percolation --')
        try:
            # Reads the percolation path report
            perc_report = pd.read_csv(self._perc_report_path)
            self.perc_dist = float(perc_report['perc_dist'].iloc[0])
            self.max_perc_disp = float(perc_report['max_perc_disp'].iloc[0])
            self.perc_calc_time = float(perc_report['perc_calc_time'].iloc[0])

            # Check if the requested percolation displacements thresholds have already been calculated.
            disp_threshold_columns = ['perc_dist_disp_{:1.1f}'.format(disp).replace('.', 'p') for disp in
                                      perc_disp_thresholds]
            if not (all([True if col in perc_report.columns else False for col in disp_threshold_columns])):
                print('\tMismatch for percolation path displacement thresholds. '
                      'Percolation path will be recalculated.')
                perc_path_recalc = True

        except FileNotFoundError:
            print(FileNotFoundError('\tPercolation path report not found. Percolation path will be calculated.'))

        except pd.errors.ParserError:
            print(pd.errors.ParserError('\tError parsing data from percolation path report. '
                                        'Percolation path will be recalculated.'))
            perc_path_recalc = True

        # Check if the percolation path files exists and perform the calculation
        if not perc_path_recalc and (os.path.isfile(self._perc_filepath) and os.path.isfile(self._perc_report_path)):
            print(f'\tPercolation path already calculated')

        else:
            self._calculate_percolation(perc_disp_thresholds=perc_disp_thresholds, **kwargs)

    def _calculate_void(self):
        """
        Calculates the distance between the grid points and the atoms in the cell structure to ascertain the position of
        the voids in the analysed structure.
        """

        print('\tCalculating void for {:s}'.format(self.structure.formula))

        start_time = time.time()

        # Get the position of the periodic images
        vertices = list(map(np.array, itertools.product([-1, 0, 1], repeat=3)))

        # Select which algorithm is used based upon the _max_vol variable, as the faster algorithm can run into memory
        # issues for cells that are too large
        if self.structure.volume < self._max_vol:
            # Get the grid point coordinates
            grid_coords = np.array([[atom.coordinate] for atom in self.grid.points])

            # Iterate over all atoms in the structure to calculate the distances
            for (n_atom, atom) in enumerate(self.structure.sites):

                # Get the position for the atom in all periodic images
                periodic_pos = np.array(
                    [np.matmul(atom.frac_coords + vert, self.grid.lattice.matrix) for vert in vertices])

                # Creates a matrix of all grid point coordinates, with a matrix shape equal to (#grid_points, #atoms)
                grid_coords_repeat = np.repeat(grid_coords, periodic_pos.shape[0], axis=1)

                # Calculates the distance and gets the minimum distance for each real atom
                distances = (periodic_pos - grid_coords_repeat) ** 2
                distances_min = np.sqrt(distances.sum(axis=2)).min(axis=1)

                # Update each grid point with a new distance to a real atom
                for (n_point, point) in enumerate(self.grid.points):
                    point.atom_dist = distances_min[n_point]

        else:
            # Iterate over all atoms in the structure to calculate the distances
            for (n_atom, atom) in enumerate(self.structure.sites):
                # Get the position for the atom in all periodic images
                periodic_pos = np.array(
                    [np.matmul(atom.frac_coords + vert, self.grid.lattice.matrix) for vert in vertices])

                # Iterate over all points in the grid and calculate their distance to one of the atoms in the structure
                for (n_point, point) in enumerate(self.grid.points):
                    distances = (periodic_pos - point.coordinate) ** 2
                    min_distance = np.sqrt(distances.sum(axis=1)).min()
                    point.atom_dist = min_distance

        final_time = time.time()
        self.time_void = final_time - start_time
        print('\tTime elapsed for void evaluation = {:.2f} min'.format(self.time_void / 60))

    def _calculate_percolation(self, perc_disp_thresholds, dist_delta: float = 0.01, perc_dist_eval: float | int = 10):

        print(f'\tCalculating percolation path for {self.structure.formula}')

        # Initializes the graph used for calculating the percolation path distance
        self._create_graph()

        # Get the number of points in each lattice vector
        npoints_a, npoints_b, npoints_c = self.grid.natoms_per_vec

        # Creates a list with the info for all the points in the grid close to the edge of the lattice cell
        # These points will serve as the initial and end points to be considered in the percolation path calculations
        start_end_points = list()
        start_end_points.extend([((self._nodes_array[0, i_b, i_c].n,
                                   self._nodes_array[-1, i_b, i_c].n),
                                  (self._nodes_array[0, i_b, i_c].atom_dist,
                                   self._nodes_array[-1, i_b, i_c].atom_dist),
                                  (self._nodes_array[0, i_b, i_c].coordinate,
                                   self._nodes_array[-1, i_b, i_c].coordinate)
                                  ) for i_b in range(npoints_b) for i_c in range(npoints_c)])
        start_end_points.extend([((self._nodes_array[i_a, 0, i_c].n,
                                   self._nodes_array[i_a, -1, i_c].n),
                                  (self._nodes_array[i_a, 0, i_c].atom_dist,
                                   self._nodes_array[i_a, -1, i_c].atom_dist),
                                  (self._nodes_array[i_a, 0, i_c].coordinate,
                                   self._nodes_array[i_a, -1, i_c].coordinate)
                                  ) for i_a in range(npoints_a) for i_c in range(npoints_c)])
        start_end_points.extend([((self._nodes_array[i_a, i_b, 0].n,
                                   self._nodes_array[i_a, i_b, -1].n),
                                  (self._nodes_array[i_a, i_b, 0].atom_dist,
                                   self._nodes_array[i_a, i_b, -1].atom_dist),
                                  (self._nodes_array[i_a, i_b, 0].coordinate,
                                   self._nodes_array[i_a, i_b, -1].coordinate))
                                 for i_a in range(npoints_a) for i_b in range(npoints_b)])
        start_end_points.sort(key=lambda x: x[0])

        # Get the initial value of the percolation distance to be evaluated in the path search algorithm
        if self.perc_dist:
            # If there is already a percolation path distance a factor equals to 10 times the dist_delta is added to it
            # to save time during the recalculation
            perc_dist_eval = self.perc_dist + dist_delta * 1E1

            # self.perc_dist is reset in order to force its recalculation
            self.perc_dist = None

        # set the variables that will be used during the percolation path calculation
        path_displacements, perc_results, percolation_shortest_path = list(), dict(), None

        start_time = time.time()
        while not self.perc_dist and perc_dist_eval > 0:
            print('\t Percolation path distance evaluated = {:.2f} A'.format(perc_dist_eval))

            # Creates a graph and start-end point pairs subset only with points which have distances to atoms higher
            # than distance being evaluated.
            sub_graph = self.graph.subgraph(
                [n for n, attrdict in self.graph.nodes.items() if attrdict['atom_dist'] > perc_dist_eval])
            sub_start_end_points = [(points, dist, coordinate) for points, dist, coordinate in start_end_points if
                                    all(True if d > perc_dist_eval else False for d in dist)]

            # Iterate over start-end point pairs
            for points, dist, coordinate in sub_start_end_points:

                # Check if a path can be established between the start-end point pair with the nodes inside the
                # subset graph
                if nx.has_path(sub_graph, *points):
                    self.perc_calc_time = time.time() - start_time

                    # Gets the information for the shortest path that can be established
                    percolation_shortest_path = nx.shortest_path(sub_graph, *points)

                    print(
                        '\tPercolation path found with distance of {:.2f} A '
                        '- {:.2f} min elapsed'.format(perc_dist_eval, self.perc_calc_time / 60))

                    # Establishes a vector between the start-end point pair for the calculation of the displacement
                    percolation_vector = np.array(coordinate[-1] - coordinate[0])

                    # Calculate the distance to the start-end point pair vector for all the points in the path found
                    for node in percolation_shortest_path:
                        path_displacements.append(np.linalg.norm(np.cross(percolation_vector,
                                                                          self._nodes_points[node].coordinate -
                                                                          coordinate[0])) / np.linalg.norm(
                            percolation_vector))

                    self.perc_dist, self.max_perc_disp = perc_dist_eval, max(path_displacements)

                    break

            # If a percolation path was not found the distance being evaluated is reduced by a delta
            perc_dist_eval -= dist_delta

        # Variables for the calculation of the percolation paths for different displacement thresholds
        perc_dist4disp_results = list()
        perc_dist4disp_eval, perc_disp_eval = self.perc_dist, self.max_perc_disp

        # Iterate over the different percolation path thresholds being investigated
        for disp_threshold in perc_disp_thresholds:
            perc_dist_found = False
            print('\tCalculating percolation distance for max displacement of {:.1f} A'.format(disp_threshold))

            while not perc_dist_found and perc_dist_eval > 0:

                # Check if the current displacement is lower or close to the threshold, the path was found
                if perc_disp_eval < disp_threshold or np.isclose(perc_disp_eval, disp_threshold, rtol=0, atol=1E-3):
                    print('\t Percolation distance for max displacement of {:.1f} A is equal to {:.2f}'.format(
                        disp_threshold, perc_dist4disp_eval))
                    perc_dist4disp_results.append(perc_dist4disp_eval)
                    perc_dist_found = True

                else:
                    print('\t Percolation path distance being evaluated = {:.2f} A'.format(perc_dist_eval))

                    # Creates a graph and start-end point pairs subset only with points which have distances to atoms bigger
                    # than distance being evaluated.
                    sub_graph = self.graph.subgraph(
                        [n for n, attrdict in self.graph.nodes.items() if attrdict['atom_dist'] > perc_dist_eval])
                    sub_start_end_points = [(points, dist, coordinate) for points, dist, coordinate in start_end_points
                                            if all(True if d > perc_dist_eval else False for d in dist)]

                    # Iterate over start-end point pairs
                    for points, dist, coordinate in sub_start_end_points:

                        # Check if a path can be established between the start-end point pair with the nodes inside the
                        # subset graph
                        if nx.has_path(sub_graph, *points):

                            # Gets the information for the shortest path that can be established
                            percolation_disp_shortest_path = nx.shortest_path(sub_graph, *points)

                            # Establishes a vector between the start-end point pair for the calculation of the
                            # displacement
                            percolation_disp_vector = np.array(coordinate[-1] - coordinate[0])

                            # Calculate the distance to the start-end point pair vector for all the points
                            # in the path found
                            path_displacements = list()
                            for node in percolation_disp_shortest_path:
                                path_displacements.append(np.linalg.norm(np.cross(percolation_disp_vector,
                                                                                  self._nodes_points[node].coordinate -
                                                                                  coordinate[0])) / np.linalg.norm(
                                    percolation_disp_vector))

                            # Updates the  distance and displacement for the evaluated percolation path
                            perc_dist4disp_eval, perc_disp_eval = perc_dist_eval, max(path_displacements)
                            break

                    # If a percolation path was not found the distance being evaluated is reduced by a delta
                    perc_dist_eval -= dist_delta

        # Add a value of 0.0 for the remaining displacement thresholds for which a path was not able to be established
        while len(perc_dist4disp_results) < len(perc_disp_thresholds):
            perc_dist4disp_results.append(0.0)

        # Creates the dictionary containing the information of the report path
        perc_results = {'perc_dist': [self.perc_dist],
                        'max_perc_disp': [self.max_perc_disp],
                        'perc_calc_time': [self.perc_calc_time]}
        for disp, dist in zip(perc_disp_thresholds, perc_dist4disp_results):
            key = 'perc_dist_disp_{:1.1f}'.format(disp)
            perc_results[key.replace('.', 'p')] = dist

        # Export the report path
        perc_report = pd.DataFrame(perc_results)
        perc_report.to_csv(self._perc_report_path, index=False)

        # Export the file for the percolation path with the highest distance to an atom in the structure
        perc_path_points = [0 if node.n not in percolation_shortest_path else node.atom_dist for node in
                            self._nodes_points]
        perc_path_points_matrix = np.array(perc_path_points).reshape(self.grid.coordinates.shape[:-1], order='F')
        perc_voldata = VolumetricData(structure=self.structure, data={'total': perc_path_points_matrix})
        perc_voldata.write_file(self._perc_filepath)

    def _create_graph(self):
        """
        Creates the graph for the calculation of the percolation paths, with the graph nodes being the point in the grid
        utilized for calculating the void space.
        """
        # Add nodes to the graph utilized for calculating the percolation paths
        for node in self._nodes_points:
            self.graph.add_node(node.n, atom_dist=node.atom_dist, pos=node.coordinate)

        # Add the edges between graph nodes following the cell vector directions
        natoms_a, natoms_b, natoms_c = self.grid.natoms_per_vec
        for i_c in range(natoms_c - 1):
            for i_b in range(natoms_b - 1):
                for i_a in range(natoms_a - 1):
                    primary_node_n = self._nodes_array[i_a, i_b, i_c].n
                    self.graph.add_edge(primary_node_n, self._nodes_array[i_a + 1, i_b, i_c].n)
                    self.graph.add_edge(primary_node_n, self._nodes_array[i_a, i_b + 1, i_c].n)
                    self.graph.add_edge(primary_node_n, self._nodes_array[i_a, i_b, i_c + 1].n)


class Grid:
    """
    Class for evenly spaced 3D grid inside a crystal cell.
    """

    def __init__(self, structure_lattice: pymatgen.core.Structure.lattice, initial_spacing: float):

        # Set the information for the 3D grid to be constructed
        self.lattice = structure_lattice
        self._initial_spacing, self._spacings = initial_spacing, None

        # Get the coordinates for the points in the 3D grid
        self.coordinates = self._get_coordinates()

        # Set the points for the 3D grid
        GridPoint.n = 0
        self.points = [GridPoint(coord) for coord in self.coordinates.reshape(-1, 3, order='F')]

    def _get_coordinates(self):
        """
        Calculates the coordinates for the evenly spaced 3D grid inside the crystal cell
        :return:
        coordinates: array containing the coordinates for all the points established for the 3D grid
        """
        # Calculate the spacing for each of the cell matrix vectors
        self._spacings = [length / np.ceil(length / self._initial_spacing) for length in self.lattice.abc]
        self.cell_spacing = [(line * self._spacings[n]) / length for n, (line, length) in
                             enumerate(zip(self.lattice.matrix, self.lattice.abc))]

        # Calculate the coordinates for the evenly spaced points
        self._spaced_vectors_points = [[spacing * i for i in range(0, int(np.floor(length / self._spacings[n])))]
                                       for n, (spacing, length) in enumerate(zip(self.cell_spacing, self.lattice.abc))]
        self.natoms_per_vec = [len(item) for item in self._spaced_vectors_points]

        coordinates = np.empty([len(i) for i in self._spaced_vectors_points] + [3])
        for n_z, z in enumerate(self._spaced_vectors_points[2]):
            for n_y, y in enumerate(self._spaced_vectors_points[1]):
                for n_x, x in enumerate(self._spaced_vectors_points[0]):
                    coordinates[n_x, n_y, n_z, :] = np.array(x + y + z)

        return coordinates


class GridPoint:
    """
    Class for a grid point in the evenly spaced 3D grid.
    """
    n = 0

    def __init__(self, coordinate: list, **kwargs):
        self._atom_dist = kwargs.get('atom_dist', np.inf)
        self.n = kwargs.get('n', GridPoint.n)
        GridPoint.n += 1
        self.coordinate = np.array(coordinate)

    @property
    def atom_dist(self):
        """
        Property class for the lowest distance from the grid point to an atom in the structure in Å

        :return:
        _atom_dist: distance from the grid point to an atom in the structure in Å
        """
        return self._atom_dist

    @atom_dist.setter
    def atom_dist(self, dist: float):
        """
        Setter for the grid point distance to an atom in the structure. The distance is only updated if the new distance
        is lower than the current one. Negative and non-numeric values are flagged as errors.

        :param dist: distance to be considered between the grid point and an atom in the structure
        """

        try:
            dist = float(dist)

        except ValueError or TypeError:
            raise TypeError(f'Distance argument must be float, not {type(dist)}')

        if dist < 0:
            raise ValueError('Provided a negative distance argument.')

        elif dist <= self.atom_dist:
            self._atom_dist = dist
