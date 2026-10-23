# Python Package TDAExplore

# Copyright (C) 2021 Parker Edwards

# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.

# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
# USA

import csv


def read_parameters(parameters):
    """Read a TDAExplore parameters csv file.

    Recognized columns include image_directories, directory_classes,
    patch_center_images_directories, radius_of_patches,
    experiment_name, and data_results_directory. A template
    parameters.csv is included at
    https://github.com/P-Edwards/TDAExplore-ML.

    Parameters
    ----------
    parameters : str
        File path to the csv file.

    Returns
    -------
    dict
        Mapping from column name to list of non-empty column values.
    """
    with open(parameters, newline="") as handle:
        reader = csv.DictReader(handle)
        columns = {name: [] for name in reader.fieldnames if name}
        for row in reader:
            for name in columns:
                value = row.get(name)
                if value is not None and value != "" and value != "NA":
                    columns[name].append(value)
    return {name: values for name, values in columns.items() if values}
