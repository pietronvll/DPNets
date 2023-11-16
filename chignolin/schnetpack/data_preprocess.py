import logging
import os
from pathlib import Path

import MDAnalysis as mda
import ml_confs
import schnetpack
from ase import Atoms
from ase.formula import Formula
from schnetpack.data import ASEAtomsData
from tqdm import tqdm

"""
configs = {
    'topology_path': '',
    'trajectory_path': '',
    'selection': '',
    'data_path': '',
}
Selection can be:
'all' - all atoms   (default)
'backbone' - backbone atoms
'not element H' - heavy atoms
"""


def trajectory_to_ase_atoms(configs: ml_confs.Configs):
    # Convert a pdb trajectory to a database of ASE atoms
    topology = configs.topology_path
    trajectory = configs.trajectory_path

    db_name = os.path.split(topology)[-1].split(".")[0]
    slug = configs.selection.replace(" ", "_")
    db_name = f"{db_name}_{slug}"

    u = mda.Universe(topology, trajectory)
    atoms_list = []
    for ts in tqdm(
        u.trajectory, total=len(u.trajectory), desc="Parsing DCD file to ASE atoms"
    ):
        selection = u.select_atoms(configs.selection)
        element_list = (u.atoms.elements)[selection.indices]
        positions = (ts.positions)[selection.indices]
        masses = (u.atoms.masses)[selection.indices]

        atoms = Atoms(
            Formula().from_list(element_list),
            positions=positions,
            cell=ts.dimensions[:3],
            pbc=True,
            masses=masses,
        )
        atoms_list.append(atoms)
    ase_db = ASEAtomsData.create(
        os.path.join(configs.data_path, f"{db_name}.db"),
        distance_unit="Ang",
        property_unit_dict={},
    )
    logging.info("Adding systems to ASE database")
    ase_db.add_systems([{}] * len(atoms_list), atoms_list=atoms_list)
    logging.info("Done!")


def preprocess_asedb(cutoff: float, db_path: os.PathLike):
    db_path = Path(db_path)
    data_path = db_path.parent
    db_name = db_path.name.split(".")[0]
    cache_path = str(data_path / f"__cache-{db_name}__")
    nb_list_transform = schnetpack.transform.CachedNeighborList(
        cache_path,
        schnetpack.transform.MatScipyNeighborList(cutoff=cutoff),
        keep_cache=True,
    )
    in_transforms = [schnetpack.transform.CastTo32(), nb_list_transform]
    dataset = schnetpack.data.ASEAtomsData(str(db_path), transforms=in_transforms)
    logging.info(f"Loaded {len(dataset)} frames from {db_path}.")
    dataloader = schnetpack.data.AtomsLoader(
        dataset, num_workers=10, persistent_workers=True
    )
    for _ in tqdm(dataloader, total=len(dataloader)):
        pass
