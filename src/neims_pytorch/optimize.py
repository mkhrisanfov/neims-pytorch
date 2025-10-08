import logging
from typing import Annotated
from pathlib import Path

import optuna
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from neims_pytorch import BASE_DIR
from neims_pytorch.datasets import get_train_test_datasets
from neims_pytorch.models import NEIMSPytorch
import typer

app = typer.Typer()


def weigthed_cosine_similarity_loss(x, y):
    masses = torch.arange(1, x.size()[1] + 1, device=x.device)
    return 1 - torch.mean(
        torch.cosine_similarity(masses * torch.sqrt(x), masses * torch.sqrt(y))
    )


@app.command()
def train(model, optim, crit, epoch_end, train_dl, test_dl, name, trial):
    torch.cuda.empty_cache()
    pbar = tqdm(range(0, epoch_end), position=0)
    b_trn = 10
    b_tst = 10
    writer = SummaryWriter(log_dir=BASE_DIR / f"logs/{name}", flush_secs=15)
    for epoch in pbar:
        train_loss = model.train_fn(optim, crit, train_dl)
        b_trn = min(b_trn, train_loss)

        test_loss = model.eval_fn(crit, test_dl)
        b_tst = min(b_tst, test_loss)

        pbar.set_postfix_str(
            f"{train_loss:.3e}/{b_trn:.3e}  {test_loss:.3e}/{b_tst:.3e}"
        )
        logging.info(
            f"[{epoch + 1}/{epoch_end}]:  trn: {train_loss:.3e}  tst: {test_loss:.3e}"
        )
        writer.add_scalar("loss/trn", train_loss, epoch)
        writer.add_scalar("loss/tst", test_loss, epoch)
        if test_loss <= b_tst:
            torch.save(model.state_dict(), BASE_DIR / f"models/{name}_model.pth")

        if trial is not None:
            trial.report(test_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    pbar.close()
    return b_trn, b_tst


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
    batch_size: Annotated[int, typer.Argument()] = 512,
):
    batch_size = batch_size
    trn_ds, tst_ds = get_train_test_datasets(input_filename)
    trn_dl = DataLoader(
        trn_ds,
        batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    tst_dl = DataLoader(
        tst_ds,
        batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    def objective(trial):
        lr = 1e-4
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        model = NEIMSPytorch(
            hidden_dim=2 ** trial.suggest_int("hidden_dim", 8, 12),
            n_middle_layers=trial.suggest_int("n_middle", 1, 8),
            n_wt_head=trial.suggest_int("n_head", 1, 5),
            n_nl_head=trial.suggest_int("n_head", 1, 5),
            n_gate_head=trial.suggest_int("n_gate_head", 1, 5),
            dropout=trial.suggest_float("dropout", 0.01, 0.25),
        ).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        crit = weigthed_cosine_similarity_loss

        b_train, b_test = train(
            model,
            optim,
            crit,
            250,
            trn_dl,
            tst_dl,
            name=f"NEIMS-optuna-{model.hidden_dim}-{model.n_middle_layers}-"
            f"{model.n_wt_head}-{model.n_gate_head}-{model.dropout:.2f}",
            trial=trial,
        )

        return b_test

    study = optuna.create_study(
        study_name="NEIMS",
        storage=f"sqlite:///{BASE_DIR / 'logs/NEIMS.db'}",
        directions=["minimize"],
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=50, n_min_trials=7
        ),
    )
    study.enqueue_trial(
        {
            "hidden_dim": 12,
            "n_middle": 2,
            "n_head": 4,
            "n_gate_head": 2,
            "dropout": 0.1,
        },
        skip_if_exists=True,
    )
    study.optimize(objective, n_trials=30)


if __name__ == "__main__":
    app()
