import re
from typing import Literal
from math import ceil

import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from neims_pytorch import MAX_MZ, SEED
from neims_pytorch.utils import generate_fingerprints


class NEIMSPytorchDataset(Dataset):
    def __init__(self, molecules, spectra=None):
        self.length = len(molecules)
        self.mol_wts = torch.FloatTensor(
            list(map(rdMolDescriptors.CalcExactMolWt, molecules))
        )
        self.fps = torch.FloatTensor(generate_fingerprints(molecules))
        if spectra is not None:
            self.spectra = torch.FloatTensor(np.vstack(spectra) / 1000)

    def __getitem__(self, index):
        if not hasattr(self, "spectra"):
            return (
                self.fps[index],
                self.mol_wts[index],
            )
        else:
            return (
                self.fps[index],
                self.mol_wts[index],
                self.spectra[index],
            )

    def __len__(self):
        return self.length


def get_train_test_datasets(filename):
    # names = []
    # smis = []
    # spectra = []
    # with open(filename, "r") as f:
    #     for line in tqdm(f):
    #         name, smiles, spectrum, _ = line.split("|")
    #         spectrum = list(map(float, spectrum.split()))
    #         dense_spectrum = np.zeros(MAX_MZ)
    #         spectrum = np.array(spectrum).reshape(-1, 2)
    #         dense_spectrum[spectrum[:, 0].astype(int)] = spectrum[:, 1]
    #         names.append(name.strip())
    #         smis.append(smiles.strip())
    #         spectra.append(dense_spectrum)

    print("Reading input file")
    compounds = read_msp(filename)
    print(f"Compounds loaded: {len(compounds)}")
    rdBase.DisableLog("rdApp.*")
    smis = [x.get("smiles", None) for x in compounds]
    inchis = [x.get("inchi", None) for x in compounds]
    spectra = [x.get("ms", None) for x in compounds]

    num_smis = np.count_nonzero(smis)
    num_inchis = np.count_nonzero(inchis)

    print("Generating Molecules")
    if num_smis == 0 and num_inchis == 0:
        raise ValueError("0 valid identifier strings found")
    inchi_mols = list(map(Chem.MolFromInchi, tqdm(inchis)))
    smiles_mols = list(map(Chem.MolFromSmiles, tqdm(smis)))
    if np.count_nonzero(inchi_mols) >= np.count_nonzero(smiles_mols):
        mols = inchi_mols
    else:
        mols = smiles_mols
    rdBase.EnableLog("rdApp.*")
    print(f"Valid molecules: {len(mols)}")

    print("Generating InChIKeys")
    inchikeys = [
        Chem.MolToInchiKey(x).split("-")[0] if x is not None else None
        for x in tqdm(mols)
    ]
    unique_inchikeys = set(inchikeys)
    if None in unique_inchikeys:
        unique_inchikeys.remove(None)
    trn_inchikeys, tst_inchikeys = train_test_split(
        sorted(list(unique_inchikeys)),  # pyright: ignore[reportArgumentType]
        test_size=0.2,
        random_state=SEED,
    )

    trn_inchikeys = set(trn_inchikeys)
    tst_inchikeys = set(tst_inchikeys)
    trn_mask, tst_mask = [], []
    for i, val in enumerate(inchikeys):
        if val in trn_inchikeys:
            trn_mask.append(i)
        elif val in tst_inchikeys:
            tst_mask.append(i)
    mols = np.array(mols)
    spectra = np.vstack(spectra)
    print("Train")
    trn_ds = NEIMSPytorchDataset(mols[trn_mask], spectra[trn_mask])

    print("Test")
    tst_ds = NEIMSPytorchDataset(mols[tst_mask], spectra[tst_mask])
    return (trn_ds, tst_ds)


def get_inference_dataset(filename, file_type: Literal["inchi", "smiles"]):
    compounds = []
    with open(filename, "r") as fin:
        for line in tqdm(fin):
            compounds.append({"id": line.rstrip()})
    if file_type == "inchi":
        for compound in compounds:
            compound["mol"] = Chem.MolFromInchi(compound["id"])
    elif file_type == "smiles":
        for compound in compounds:
            compound["mol"] = Chem.MolFromSmiles(compound["id"])
    else:
        raise ValueError(
            "Unsupported file type. Provide a list of either SMILES or InChI."
        )
    print("Populating dataset")
    rdBase.DisableLog("rdApp.warning")
    for compound in compounds:
        if compound["mol"] is None:
            compound["inchi"] = None
            compound["inchikey"] = None
            compound["smiles"] = None
        else:
            compound["inchi"] = Chem.MolToInchi(compound["mol"])
            compound["inchikey"] = Chem.MolToInchiKey(compound["mol"])
            compound["smiles"] = Chem.MolToSmiles(compound["mol"])

    dataset = NEIMSPytorchDataset(
        [compound["mol"] for compound in compounds if compound["mol"] is not None],
        spectra=None,
    )
    return compounds, dataset


def read_msp(filename):
    compounds = []
    compound = {"ms": np.zeros(MAX_MZ)}
    pattern = re.compile(r"^(?P<key>[\d\w\s]+):\s+(?P<val>.+)$")
    ms_pattern = re.compile(r"^(?P<mz>[\d\.]+)\s+(?P<int>\d+)$")
    with open(filename, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            match = re.match(pattern, line)
            ms_match = re.match(ms_pattern, line)
            if match:
                compound[match["key"]] = match[  # pyright: ignore[reportArgumentType]
                    "val"
                ]
            elif int(compound.get("num peaks", 0)) > 0 and ms_match:
                mz = int(ceil(float(ms_match["mz"]) - 0.65))
                if mz < MAX_MZ:
                    compound["ms"][mz] = float(ms_match["int"])
            elif line == "\n":
                compound["ms"] = (
                    compound["ms"]
                    / np.clip(compound["ms"].max(), a_min=1e-8, a_max=None)
                    * 999
                )
                compounds.append(compound)
                compound = {"ms": np.zeros(MAX_MZ)}
    return compounds
