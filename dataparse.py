# Copyright 2026 Caio Miranda Miliante

import pandas as pd
from mp_api.client import MPRester
import json
import os
import time
import mysql.connector
from pymatgen.core import Structure
from pymatgen.symmetry import analyzer
import numpy as np
from void import Void
from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram, PourbaixPlotter
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms

matplotlib.use('Agg')

# Base path for materials data to be stored
MATDATA_PATH = os.path.join(os.getcwd(), 'matdata')

# Personal materials project API key. Can be retrieved from https://next-gen.materialsproject.org/api
MP_API_KEY = ''  # To be added by user

# Connector for local MySQL server
MYDB = mysql.connector.connect(
    host='localhost',
    user='root',
    # Personal password for connecting to locally hosted MySQL server
    passwd='')  # To be added by user
MYCURSOR = MYDB.cursor()

# Database name for the project
database_name = 'IntercalationSearch' 

# Check if the database for the project exists, and if not it creates it
MYCURSOR.execute('CREATE DATABASE IF NOT EXISTS {:s}'.format(database_name))
MYCURSOR.execute('USE {:s}'.format(database_name))


def mp_query(chemsys_groups: list, **kwargs):
    # Inherit the global variables to be utilized in the function
    global MATDATA_PATH, MP_API_KEY, MYDB, MYCURSOR

    # By default, only structures confirmed to be experimentally obtained are going to be parsed by the code
    only_exp = kwargs.get('only_exp', True)

    # Variable for recalculating the compound data. By default, it is False
    mat_recalc = kwargs.get('mat_recalc', False)

    # Priority order with respect to XC functional utilized in calculation for thermodynamic data retrieval
    thermo_types = ['GGA_GGA+U_R2SCAN', 'R2SCAN', 'GGA_GGA+U']

    # Creation of the SQL table for storing the data
    table_name = kwargs.get('table_name', 'chemical_info')
    table_dict = {'mp_id': 'INT PRIMARY KEY NOT NULL',  # The Materials Project ID is the primary key for distinguishing
                  # between materials investigated
                  'formula': 'VARCHAR(64) NOT NULL',
                  'formula_unit': 'VARCHAR(64) NOT NULL',
                  'elements': 'VARCHAR(64) NOT NULL',
                  'volume': 'DECIMAL(12,3) NOT NULL',
                  'energy_above_hull': 'DECIMAL(11,8) NOT NULL',
                  'thermo_type': 'VARCHAR(64) NOT NULL',
                  'experimental': 'TINYINT(1) NOT NULL',
                  'avg_ox_state': 'VARCHAR(64)',
                  'decomposes_to': 'VARCHAR(255)',
                  'dir_path': 'VARCHAR(255) NOT NULL',
                  'json_path': 'VARCHAR(255) NOT NULL'}
    MYCURSOR.execute('CREATE TABLE IF NOT EXISTS {:s} ({:s})'.format(table_name, ', '.join(' '.join(item) for item in
                                                                                           table_dict.items())))

    # Check for materials that already have had their data retrieved
    MYCURSOR.execute('SELECT mp_id FROM {:s}'.format(table_name))
    mp_ids_table = [item for items in MYCURSOR.fetchall() for item in items]

    # Loops through all the chemical systems provided by to the function
    for chemsys in chemsys_groups:
        initial_time = time.time()

        # Initializes the connection to the MP via its API using the key provided by the user
        with MPRester(MP_API_KEY) as mpr:
            print(f'Searching for {chemsys}.')

            # Query all compounds for the given chemsys
            compounds = mpr.materials.search(chemsys=chemsys)
            if not any(compounds):
                continue

            # Filter compounds list for only experimental values if only_exp == True
            if only_exp:
                compounds_id = [compound.material_id for compound in compounds]
                exps = {compound.material_id: not compound.theoretical for compound in
                        mpr.materials.summary.search(material_ids=compounds_id, fields=["material_id", "theoretical"])}
                compounds = [compound for compound in compounds if exps[compound.material_id]]

            # Filter out compounds that are already in the SQL table
            if not mat_recalc:
                compounds = [compound for compound in compounds if
                             int(str(compound.material_id).split('-')[-1]) not in mp_ids_table]

            # Loop for data retrieval of compounds to be investigated
            for compound in compounds:

                # Get the path for where the material data will be stored
                compound_path = os.path.join(MATDATA_PATH, str(compound.material_id).split('-')[-1])
                if not os.path.isdir(compound_path):
                    os.makedirs(compound_path)

                # Get path to json file and dump the .json file extracted from the MP to the assigned path
                compound_path_json = os.path.join(compound_path, str(compound.material_id) + '.json')
                with open(compound_path_json, 'w') as f:
                    json.dump(compound.structure.as_dict(), f)

                # Get the information with respect to the conventional structure
                mat_analyzer = analyzer.SpacegroupAnalyzer(compound.structure)
                mat_struct = mat_analyzer.get_conventional_standard_structure()

                # Query thermodynamical, oxidation and experimental information for compound
                try:
                    # Search for thermodynamic data with respect to the thermo_types in order, only retrieving the data
                    # for the result with the highest thermodynamic priority
                    compound_thermo = mpr.thermo.search(material_ids=[compound.material_id],
                                                        thermo_types=thermo_types)[0]
                    compound_ox = mpr.oxidation_states.search(material_ids=[compound.material_id])[0]
                    compound_exp = mpr.summary.search(material_ids=[compound.material_id], fields=['theoretical'])[0]
                except IndexError:
                    print(f'Thermodynamic query error for compound {compound.material_id}')
                    continue

                # Clean decomposes_to information
                if compound_thermo.decomposes_to:
                    decomposes_to = ' + '.join(['{:1.2f}'.format(mat.amount) + ''.join(mat.formula.split())
                                                for mat in compound_thermo.decomposes_to])
                else:
                    decomposes_to = 'None'

                # Get compound data in a dictionary, correcting order for incorporating in the table
                compound_data = {'mp_id': int(str(compound.material_id).split('-')[-1]),
                                 'formula': ''.join(compound.structure.formula.split()),
                                 'formula_unit': compound.formula_pretty,
                                 'elements': '-'.join([str(ele) for ele in compound.elements]),
                                 'volume': mat_struct.volume,
                                 'energy_above_hull': compound_thermo.energy_above_hull,
                                 'thermo_type': str(compound_thermo.thermo_type),
                                 'experimental': 0 if compound_exp.theoretical else 1,
                                 'avg_ox_state': '-'.join(['{}({:.2f})'.format(str(ele), float(ox)) for ele, ox in
                                                           compound_ox.average_oxidation_states.items()]),
                                 'decomposes_to': str(decomposes_to),
                                 'dir_path': os.path.relpath(compound_path).replace(os.sep, '/'),
                                 'json_path': os.path.relpath(compound_path_json).replace(os.sep, '/')}

                # Insert queried data information into the SQL table
                if compound_data['mp_id'] in mp_ids_table:
                    query_values = ', '.join(['{:s}=\'{:s}\''.format(key, str(value)) for key, value in
                                              compound_data.items() if key != 'mp_id'])

                    MYCURSOR.execute('UPDATE {:s} SET {:s} WHERE {:s}'.format(table_name, query_values,
                                                                              'mp_id={:d}'.format(
                                                                                  compound_data['mp_id'])))
                else:
                    columns = ', '.join(list(map(str, compound_data.keys())))
                    values = ', '.join(["'{}'".format(str(val)) for val in compound_data.values()])
                    MYCURSOR.execute('INSERT INTO {:s} ({:s}) VALUES ({:s})'.format(table_name, columns, values))

                MYDB.commit()

            final_time = time.time()
            print('Total elapsed time =  {:.2f} minutes.\n'.format((final_time - initial_time) / 60))


def void_screening(input_table_name: str, **kwargs):
    # Inherit the global variables to be utilized in the function
    global MYDB, MYCURSOR

    # togle if the void calculation will be rechecked for materials already present on a previously existent mp_id table
    void_recheck = kwargs.get('void_recheck', True)

    # Void distances to have the unit cells evaluated for with respect to its percentages
    void_distances = kwargs.get('void_distances', np.arange(.5, 7, .5))

    # Percolation path displacements values to be considered for when evaluating the highest percolation paths distances
    # that can be established in the cell for each displacement value
    perc_disp_thresholds = kwargs.get('perc_disp_thresholds', np.linspace(1, 0, 11))

    # Create SQL table which will contain the results of void and percolation path screening
    out_table_name = kwargs.get('out_table_name', 'void_screening')
    out_table_dict = {'mp_id': 'INT PRIMARY KEY NOT NULL',
                      'perc_dist': 'DECIMAL(9,6) NOT NULL',
                      'max_perc_disp': 'DECIMAL(9,6) NOT NULL',
                      'perc_calc_time': 'DECIMAL(12,4) NOT NULL',
                      'time_void': 'DECIMAL(12,4) NOT NULL'}
    out_table_dict.update({'void_perc_{:1.1f}'.format(dist).replace('.', 'p'): 'DECIMAL(10,8) NOT NULL'
                           for dist in void_distances})
    out_table_dict.update({'perc_dist_disp_{:1.1f}'.format(disp).replace('.', 'p'): 'DECIMAL(6,4) NOT NULL'
                           for disp in perc_disp_thresholds})
    MYCURSOR.execute('CREATE TABLE IF NOT EXISTS {:s} ({:s})'.format(out_table_name, ', '.join(' '.join(item)
                                                                                               for item in
                                                                                               out_table_dict.items())))

    # Check for materials that already have had their void and percolation path distance calculated
    MYCURSOR.execute('SELECT mp_id FROM {:s}'.format(out_table_name))
    out_mp_ids = [item for items in MYCURSOR.fetchall() for item in items]

    # Confirm that the input table provided exists and retrieves relevant data from the input table
    MYCURSOR.execute('SHOW TABLES LIKE %s;', (input_table_name,))
    tables_avail = [item for items in MYCURSOR.fetchall() for item in items]
    if input_table_name not in tables_avail:
        raise FileNotFoundError('Table requested ({:s}) does not exist in provided database'.format(input_table_name))
    MYCURSOR.execute('SELECT mp_id, formula_unit, dir_path, json_path FROM {:s}'.format(input_table_name))
    materials_data = MYCURSOR.fetchall()

    print('Performing void screening for materials.')

    # Iterate over the materials in the input table to perform the void and percolation path calculation in them
    for n, (mp_id, formula_unit, dir_path, json_path) in enumerate(materials_data):
        print('{:s}-mp_{:d}'.format(formula_unit, mp_id))

        # Check if the data for the material in question has already been calculated or if a recheck is asked for
        if void_recheck or mp_id not in out_mp_ids:

            # Reads the json file
            try:
                with open(json_path, 'r') as f:
                    mat_json = json.load(f)
            except FileNotFoundError:
                FileNotFoundError('.json file for {:d} not found.'.format(mp_id))

            # Read information from json file, perform void analysis and get the void info
            mat_analyzer = analyzer.SpacegroupAnalyzer(Structure.from_dict(mat_json))
            mat_struct = mat_analyzer.get_conventional_standard_structure()
            mat_va = Void(mat_struct, dir_path, **kwargs)
            mat_va.percolation(perc_disp_thresholds=perc_disp_thresholds)
            void_info = mat_va.void_info(dists_eval=np.arange(.5, 7, .5))
            perc_info = mat_va.perc_info()
            data_info = {'mp_id': mp_id, **void_info, **perc_info}

            # Export data to SQL table
            if data_info['mp_id'] in out_mp_ids:
                query_values = ', '.join(
                    ['{:s}=\'{:s}\''.format(key, str(value)) for key, value in data_info.items()
                     if key != 'mp_id'])
                MYCURSOR.execute('UPDATE {:s} SET {:s} WHERE {:s}'.format(out_table_name, query_values,
                                                                          'mp_id={:d}'.format(data_info['mp_id'])))

            else:
                columns = ', '.join(list(map(str, data_info.keys())))
                values = ', '.join(["'{}'".format(str(val)) for val in data_info.values()])

                MYCURSOR.execute('INSERT INTO {:s} ({:s}) VALUES ({:s})'.format(out_table_name, columns, values))

            MYDB.commit()


def pourbaix_analysis(input_table_name: str, recalculate: bool = False, **kwargs):
    
    global MP_API_KEY, MYDB, MYCURSOR

    def pourbaix_plot_query(ax_plot, dir_path):

        plot_path = os.path.join(dir_path, f'PourbaixDiagram-{int(entry.entry_id.split("-")[1])}.png')
        plot_path_clean = os.path.join(dir_path,
                                       f'PourbaixDiagram-{int(entry.entry_id.split("-")[1])}-clean.png')

        if not os.path.isfile(plot_path) or not os.path.isfile(plot_path_clean):

            for child in ax_plot.get_children():
                if isinstance(child, matplotlib.text.Annotation):
                    child.set_color('Black')
                    x, y = child.get_position()
                    child.set_position((x, y + 0.763))
                    child.set_fontsize(22)

                elif isinstance(child, matplotlib.collections.PolyCollection):

                    transoffset = matplotlib.transforms.Affine2D().translate(*np.array([0, 0.763]))
                    child.set_transform(transoffset + child.get_transform())

                elif isinstance(child, matplotlib.lines.Line2D):
                    y = child.get_ydata()
                    child.set_ydata(np.array(y) + 0.763)
                    if child.get_ls() == '--':
                        child.set_color('White')
                        child.set_alpha(.85)

            # Adjusting the Colorbar

            ax_plot.collections[-1].colorbar.set_label(f"Stability of {entry.composition} (eV/atom)",
                                                       fontsize=16)
            ax_plot.collections[-1].colorbar.outline.set_linewidth(2.5)
            ax_plot.collections[-1].colorbar.ax.tick_params(length=6, width=2, labelsize=16)

            ax_plot.axis(np.array(ax_plot.get_xbound() + ax_plot.get_ybound()) + np.array([0, 0, 0.763, 0.763]))
            ax_plot.tick_params(axis='both', which='major', direction='in', length=6, width=2, right=True, top=True,
                                labelsize=20)
            ax_plot.tick_params(axis='both', which='minor', direction='in', length=4, width=2, right=True, top=True)

            ax_plot.set_ylabel('E (V vs Zn/Zn$^{2+}$)', fontsize=26)
            ax_plot.set_xlabel('pH', fontsize=26)

            for axis in ['top', 'bottom', 'left', 'right']:
                ax_plot.spines[axis].set_linewidth(2.5)
                ax_plot.spines[axis].set_color("Black")

            plt.savefig(plot_path, dpi=100, bbox_inches='tight')

            for child in ax_plot.get_children():
                if isinstance(child, matplotlib.text.Annotation):
                    child.remove()

            plt.savefig(plot_path_clean, dpi=100, bbox_inches='tight')
            plt.close()

    ele_conc = kwargs.get('ele_conc', 1E-3)
    ph_range = kwargs.get('ph_range', [4, 6])
    pot_range = kwargs.get('pot_range', [0, 1.229])  # Potential range at pH = 0. 0 is HER and 1.229 is OER

    # Create all the points for the pH to be analyzed in
    base_phs, base_potentials, ph_pots = np.linspace(*ph_range, 20), np.linspace(*pot_range, 20), list()
    for ph in base_phs:
        ph_pots.extend([(ph, pot) for pot in (base_potentials - 0.059159343 * ph)])

    # Create the table for outputing the Pourbaix results if it does not exist already
    out_table_name = 'pourbaix_analysis'
    out_table_dict = {'mp_id': 'INT PRIMARY KEY NOT NULL',
                      'pourbaix_stab_avg': 'DECIMAL(10,7) NULL',
                      'pourbaix_stab_median': 'DECIMAL(10,7) NULL',
                      'pourbaix_stab_std': 'DECIMAL(10,7)  NULL',
                      'pourbaix_stab_avg_min': 'DECIMAL(10,7) NULL',
                      'pourbaix_stab_avg_max': 'DECIMAL(10,7) NULL',
                      'pourbaix_stab_min': 'DECIMAL(10,7) NULL',
                      'pourbaix_stab_max': 'DECIMAL(10,7) NULL'}

    MYCURSOR.execute('CREATE TABLE IF NOT EXISTS {:s} ({:s})'.format(out_table_name, ', '.join(' '.join(item)
                                                                                               for item in
                                                                                               out_table_dict.items())))

    # Get the mp_ids for which the Pourbaix stability was already calculated
    MYCURSOR.execute('SELECT mp_id FROM {:s}'.format(out_table_name))
    out_mp_ids = [item for items in MYCURSOR.fetchall() for item in items]

    # Read csv file with the materials that will have their Pourbaix diagram plotted
    MYCURSOR.execute('SHOW TABLES LIKE %s;', (input_table_name,))
    tables_avail = [item for items in MYCURSOR.fetchall() for item in items]

    if input_table_name not in tables_avail:
        raise FileNotFoundError('Table requested ({:s}) does not exist in provided database'.format(input_table_name))

    # Create a dictionary with the different chemsys as key and the values being the df with each material that has the
    # respective chemsys. The material is not included if the mp_id is already in the output_table.
    input_table_df = pd.read_sql('SELECT * FROM {:s}'.format(input_table_name), MYDB)

    if not recalculate:
        input_table_df = input_table_df.loc[~input_table_df['mp_id'].isin(out_mp_ids)]

    materials_per_chemsys = {chemsys: input_table_df.loc[input_table_df['elements'] == chemsys] for chemsys in
                             set(input_table_df['elements'])}

    # Connect to Materials Project API and retrieve the materials Pourbaix Diagram
    with MPRester(api_key=MP_API_KEY) as mpr:

        # Perform the Pourbaix Analysis info for every chemsys and plot the diagram on the correspondent directory
        for chemsys, chemsys_df in materials_per_chemsys.items():
            print(f'Analyzing Pourbaix Stability for {chemsys} materials.')

            base_entries = mpr.get_pourbaix_entries(chemsys=chemsys)
            pbx_entries = {mp_id: [] for mp_id in list(chemsys_df['mp_id'])}
            pbx_entries.update({int(e.entry_id.split('-')[1]): e for e in base_entries if
                                int(e.entry_id.split('-')[1]) in list(chemsys_df['mp_id'])})

            # Do the Pourbaix diagram analysis
            for mp_id_entry, entry in pbx_entries.items():

                if entry:
                    print(f'{entry.composition}\t{entry.entry_id}')

                    # Get the entry information for the PourbaixDiagram function and initiates it
                    comp_dict = {str(ele): entry.composition.get(ele) for ele in entry.composition.elements
                                 if str(ele) not in ['O', 'H']}
                    total_atoms = sum(comp_dict.values())
                    comp_dict = {key: value / total_atoms for key, value in comp_dict.items()}
                    pbx = PourbaixDiagram(base_entries, comp_dict=comp_dict,
                                          conc_dict={key: ele_conc for key in comp_dict.keys()})
                    plotter = PourbaixPlotter(pbx)
                    ax_ = plotter.plot_entry_stability(entry, label_fontsize=18,
                                                       show_water_lines=True, show_neutral_axes=False)
                    dir_path = chemsys_df[chemsys_df['mp_id'] == int(entry.entry_id.split('-')[1])].iloc[0]['dir_path']
                    pourbaix_plot_query(ax_, dir_path)

                    # Calculcate the stability value for all the ph and potential points in ph_pots
                    entry_stabilities = sorted(
                        [pbx.get_decomposition_energy(entry, pH=ph, V=pot) for ph, pot in ph_pots])

                    # Calculate the statistically relevant values
                    entry_average, entry_median, entry_std = np.average(entry_stabilities), np.median(
                        entry_stabilities), \
                        np.std(entry_stabilities)
                    entry_avgmax, entry_avgmin = entry_average + entry_std, entry_average - entry_std
                    entry_min, entry_max = min(entry_stabilities), max(entry_stabilities)

                    # Write the statistics to dictionary
                    entry_stats = {'mp_id': int(entry.entry_id.split('-')[1]),
                                   'pourbaix_stab_avg': entry_average,
                                   'pourbaix_stab_median': entry_median,
                                   'pourbaix_stab_std': entry_std,
                                   'pourbaix_stab_avg_min': entry_avgmin,
                                   'pourbaix_stab_avg_max': entry_avgmax,
                                   'pourbaix_stab_min': entry_min,
                                   'pourbaix_stab_max': entry_max}

                else:
                    # Write the statistics to dictionary
                    entry_stats = {'mp_id': int(mp_id_entry)}

                if entry_stats['mp_id'] in out_mp_ids:
                    query_values = ', '.join(
                        ['{:s}=\'{:s}\''.format(key, str(value)) for key, value in entry_stats.items()
                         if key != 'mp_id'])
                    MYCURSOR.execute('UPDATE {:s} SET {:s} WHERE {:s}'.format(out_table_name, query_values,
                                                                              'mp_id={:d}'.format(
                                                                                  entry_stats['mp_id'])))

                else:
                    columns = ', '.join(list(map(str, entry_stats.keys())))
                    values = ', '.join(["'" + str(val) + "'" for val in entry_stats.values()])

                    MYCURSOR.execute('INSERT INTO {:s} ({:s}) VALUES ({:s})'.format(out_table_name, columns, values))

            print('\n')

            MYDB.commit()
