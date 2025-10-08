from typing import Annotated, Literal
from pathlib import Path

import numpy as np
import torch
import typer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from neims_pytorch import BASE_DIR
from neims_pytorch.datasets import (
    get_inference_dataset,
)
from neims_pytorch.models import NEIMSPytorch

app = typer.Typer()


def spec_to_string(array, threshold: float = 10):
    out = ""
    for i, spec in enumerate(array):
        if spec > threshold:
            out += f"{i} {round(spec):d}\n"
    return out


@app.command()
def main(
    input_filename: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    input_type: Annotated[Literal["inchi", "smiles"], typer.Argument()],
    output_filename: Annotated[
        Path,
        typer.Argument(
            exists=False,
            resolve_path=True,
        ),
    ],
    model_weigths: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = BASE_DIR
    / "models/NEIMSPyTorch.pth",
    batch_size: Annotated[int, typer.Argument()] = 64,
    use_cuda: Annotated[bool, typer.Option("--use_cuda")] = False,
):
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        typer.echo("Using CUDA")
    else:
        device = torch.device("cpu")
        typer.echo("Using CPU")

    typer.echo("Loading dataset")
    compounds, fw_dataset = get_inference_dataset(input_filename, input_type)
    typer.echo(f"Molecules loaded: {len(compounds)}")
    typer.echo(f"Valid molecules: {len(fw_dataset)}")
    fw_dataloader = DataLoader(
        fw_dataset,
        batch_size,
        num_workers=4 if use_cuda else False,
        pin_memory=True if use_cuda else False,
        shuffle=False,
    )

    typer.echo("Loading NEIMS-PyTorch model")
    model = NEIMSPytorch(
        hidden_dim=2**12,
        n_middle_layers=2,
        n_wt_head=4,
        n_nl_head=4,
        n_gate_head=2,
        dropout=0.10,
    ).to(device)
    model.load_state_dict(
        torch.load(
            model_weigths,
            map_location=device,
            weights_only=True,
        )
    )

    typer.echo("Begin prediction")
    predicted_spectra = model.predict(fw_dataloader)

    typer.echo("Writing output file")
    with open(output_filename, "w") as fout:
        j = 0
        for i, compound in enumerate(tqdm(compounds)):
            fout.write(f"name: {compound['id'].rstrip('\n')}\n")
            fout.write(f"inchi: {compound['inchi']}\n")
            fout.write(f"inchikey: {compound['inchikey']}\n")
            fout.write(f"smiles: {compound['smiles']}\n")

            if compound["mol"] is None:
                fout.write("num peaks: 0\n")
                fout.write("0 0\n")
                fout.write("\n")
            else:
                num_peaks = np.count_nonzero(predicted_spectra[j] > 10)
                fout.write(f"num peaks: {num_peaks}\n")
                fout.write(spec_to_string(predicted_spectra[j], 10))
                fout.write("\n")
                j += 1


if __name__ == "__main__":
    app()
