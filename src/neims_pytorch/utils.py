import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from tqdm.contrib.concurrent import thread_map
from neims_pytorch import NL_ADD_MASS, RADIUS


def generate_fingerprints(molecules):
    print("Morgan fingerprints")
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS, fpSize=1024
    )
    morgan_fingerprints = np.array(
        thread_map(
            morgan_generator.GetCountFingerprintAsNumPy,
            molecules,
            chunksize=500,
            max_workers=8,
        )
    )

    print("RDKit fingerprints")
    rdkit_generator = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=1024)
    rdkit_fingerprints = np.array(
        thread_map(
            rdkit_generator.GetCountFingerprintAsNumPy,
            molecules,
            chunksize=500,
            max_workers=8,
        )
    )
    return np.hstack([morgan_fingerprints, rdkit_fingerprints])


def get_mol_wt(smiles):
    mol_wt = rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(smiles))
    return mol_wt


def to_nl_spectra(spectra, smis):
    spectra = np.array(spectra)
    if len(np.array(spectra).shape) == 1:
        spectra = np.array(spectra)[np.newaxis]
    spec = np.zeros_like(spectra, dtype=np.float64)
    wts = np.sqrt(spectra) * np.arange(1, spectra.shape[1] + 1)
    n_spec = wts / np.sum(np.square(wts), axis=1)[:, np.newaxis]
    # n_spec = spectra
    for i in range(len(smis)):
        mol_wt = get_mol_wt(smis[i])
        nonzero_idx = spectra[i].nonzero()[0]
        spec_nz = np.clip(
            np.ceil(mol_wt - 0.65) + NL_ADD_MASS - nonzero_idx,
            a_min=-1,
            a_max=spectra.shape[1] - 1,
        ).astype(int)
        spec[i, spec_nz] = n_spec[i, nonzero_idx]
    spec[:, -1] = 0
    return spec


def to_og_spectra(spectra, smis=None):
    if len(np.array(spectra).shape) == 1:
        spectra = np.array(spectra)[np.newaxis]
    if smis is not None:
        orig_spec = np.zeros_like(spectra, dtype=np.float64)
        for i in range(len(smis)):
            mol_wt = get_mol_wt(smis[i])
            nonzero_idx = spectra[i].nonzero()[0]
            orig_mz = np.clip(
                np.ceil(mol_wt - 0.65) + NL_ADD_MASS - nonzero_idx,
                a_min=-1,
                a_max=spectra.shape[1] - 1,
            ).astype(int)
            orig_spec[i, orig_mz] = spectra[i, nonzero_idx]
        spectra = orig_spec
        spectra[:, -1] = 0
    idx = np.arange(1, spectra.shape[1] + 1)
    denom = np.sum(spectra**2, axis=1, keepdims=True)
    orig_spectra = (spectra**2) / ((idx[np.newaxis, :] ** 2) * (denom**2))
    return np.round(orig_spectra / np.max(orig_spectra, axis=1)[:, np.newaxis] * 999, 0)


def nl_to_wt(nl_spectra, smis):
    if len(np.array(nl_spectra).shape) == 1:
        nl_spectra = np.array(nl_spectra)[np.newaxis]
    norm_spec = np.zeros_like(nl_spectra, dtype=np.float64)
    for i in range(len(smis)):
        mol_wt = get_mol_wt(smis[i])
        nonzero_idx = nl_spectra[i].nonzero()[0]
        orig_mz = np.clip(
            np.ceil(mol_wt - 0.65) + NL_ADD_MASS - nonzero_idx,
            a_min=-1,
            a_max=nl_spectra.shape[1],
        ).astype(int)
        # orig_mz = orig_mz[orig_mz >= 0]
        norm_spec[i, orig_mz] = nl_spectra[i, nonzero_idx]
    norm_spec[:, -1] = 0
    return norm_spec


def to_wt_spectra(spectra):
    spectra = np.array(spectra)
    if len(spectra.shape) == 1:
        spectra = spectra[np.newaxis]
    wts = np.sqrt(spectra) * np.arange(1, spectra.shape[1] + 1)
    return wts / np.sum(np.square(wts), axis=1)[:, np.newaxis]


def to_og_torch(spectra, mol_wts=None):
    if spectra.ndim == 1:
        spectra = spectra.unsqueeze(0)
    if mol_wts is not None:
        nonzero_mask = spectra != 0
        batch_idx, nonzero_idx = nonzero_mask.nonzero(as_tuple=True)
        vals = spectra[batch_idx, nonzero_idx]
        shift = torch.ceil(mol_wts[batch_idx] - 0.65) + NL_ADD_MASS
        orig_mz = torch.clamp(
            shift - nonzero_idx, min=-1, max=spectra.size(1) - 1
        ).int()
        wt_spec = torch.zeros_like(spectra, dtype=torch.float, device=spectra.device)
        wt_spec[batch_idx, orig_mz] = vals
        wt_spec[:, -1] = 0
    else:
        wt_spec = spectra
    idx = torch.arange(1, wt_spec.size(1) + 1, device=spectra.device)
    denom = torch.sum(wt_spec**2, dim=1, keepdim=True)
    orig_spectra = (wt_spec**2) / ((idx[None, :] ** 2) * (denom**2))
    return torch.round(
        orig_spectra / torch.max(orig_spectra, dim=1, keepdim=True)[0] * 999,
        decimals=0,
    )


def nl_to_wt_torch(nl_spectra, mol_wts):
    nonzero_mask = nl_spectra != 0
    nz = nonzero_mask.nonzero()
    batch_idx, nonzero_idx = nz[:, 0], nz[:, 1]
    vals = nl_spectra[batch_idx, nonzero_idx]
    # here mol_wts should use batch_idx to avoid errors
    shift = torch.ceil(mol_wts[batch_idx] - 0.65) + NL_ADD_MASS
    orig_mz = torch.clamp(shift - nonzero_idx, min=-1, max=nl_spectra.size(1) - 1).int()
    norm_spec = torch.zeros_like(
        nl_spectra,
        dtype=torch.float,
        device=nl_spectra.device,
    )
    # causes issues when batch_idx is used in a wrong way
    norm_spec[batch_idx, orig_mz] = vals
    norm_spec[:, -1] = 0
    return norm_spec


# @torch.jit.script
def nl_to_og_torch(spectra, mol_wts):
    nonzero_mask = spectra != 0
    nz = nonzero_mask.nonzero()
    batch_idx, nonzero_idx = nz[:, 0], nz[:, 1]
    vals = spectra[batch_idx, nonzero_idx]
    shift = torch.ceil(mol_wts[batch_idx] - 0.65) + NL_ADD_MASS
    orig_mz = torch.clamp(shift - nonzero_idx, min=-1, max=spectra.size(1) - 1).int()
    wt_spec = torch.zeros_like(spectra, dtype=torch.float, device=spectra.device)
    wt_spec[batch_idx, orig_mz] = vals
    wt_spec[:, -1] = 0
    idx = torch.arange(1, wt_spec.size(1) + 1, device=spectra.device)
    denom = torch.clamp(torch.sum(wt_spec**2, dim=1, keepdim=True), min=1e-8)
    orig_spectra = (wt_spec**2) / ((idx[None, :] ** 2) * (denom**2))
    return orig_spectra / torch.clamp(
        torch.max(orig_spectra, dim=1, keepdim=True)[0], min=1e-8
    )
