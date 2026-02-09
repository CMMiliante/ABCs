ABCs: Aqueous Battery Cathode Screening
================================================

## About 
ABCs (**A**queous **B**attery **C**athode **s**creening) is a Python package for the screening of novel cathode materials for aqueous ion batteries via structural and electrochemical stability analysis.

The current version of this package screens for:
-	**Structural availability for reversible ion intercalation** - The code quantifies the available empty voids and ionic percolation paths in a crystal cell, allowing for the identification of materials with stable reversible ionic intercalation during battery cycling.
-	**Electrochemical stability in aqueous media** - The code evaluates the electrochemical decomposition energy for a material under a considered battery potential range and pH, allowing for the quantification of the electrochemical stability in aqueous electrolyte under battery cycling conditions. 

The current version of the package retrieves material data from the [Materials Project database](https://next-gen.materialsproject.org/), with the extracted data and screened properties results stored in a locally hosted SQL database. The user’s personal Materials Project and SQL credentials should be entered in the ‘dataparse.py’ code in order to make use of the code. The code can be easily adjusted by the user to retrieve data from other publicly available databases, as well as to change the data management protocol.

This code has been successfully implemented for the discovery of novel cathode materials for rechargeable aqueous zinc-ion batteries, with the associated manuscript available on [arXiv](arxiv.link)

## Usage
The following code snippet serves as an example for the screening of binary metal oxide structures (*M*<sub>*x*</sub>O<sub>*y*</sub>, where M is a transition metal) as cathode materials for aqueous ion batteries.

```
from dataparse import mp_query, void_screening, pourbaix_analysis

# List of transition metals
transition_metals = ['Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Mo', 'W']

# Chemical Systems to be considered
chemsys_query = ['-'.join([metal, 'O']) for metal in transition_metals])

# Querying the chemsys for extracting experimentally confirmed materials from the Materials Project database
mp_query(chemsys_query, table_name='chemical_info', only_exp=True)

# Calculation of available voids and ion percolation paths in the structures present in the 'chemical_info' table
void_screening('chemical_info')

# Performs the electrochemical stability analysis for the materials present in the 'chemical_info' table 
pourbaix_analysis('chemical_info')

```



 
