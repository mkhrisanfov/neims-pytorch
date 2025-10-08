import torch
from torch import nn
from neims_pytorch.utils import nl_to_og_torch
from tqdm.auto import tqdm
from neims_pytorch import MAX_MZ


class NEIMSPytorch(nn.Module):
    def __init__(
        self, hidden_dim, n_middle_layers, n_wt_head, n_nl_head, n_gate_head, dropout
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_middle_layers = n_middle_layers
        self.n_wt_head = n_wt_head
        self.n_nl_head = n_nl_head
        self.n_gate_head = n_gate_head
        self.dropout = dropout
        self.fc_in = nn.Sequential(nn.Linear(2048, hidden_dim), nn.SiLU())

        self.fc_middle_layers = nn.ModuleList(
            [
                self._make_lin_layer(hidden_dim, hidden_dim, dropout)
                for _ in range(n_middle_layers)
            ]
        )
        self.fc_wt_head = nn.ModuleList(
            [
                self._make_lin_layer(hidden_dim, hidden_dim, dropout)
                for _ in range(n_wt_head)
            ]
        )
        self.fc_wt_head.append(
            nn.Sequential(
                nn.Linear(hidden_dim, MAX_MZ),
                nn.Identity(),
            )
        )

        self.fc_nl_head = nn.ModuleList(
            [
                self._make_lin_layer(hidden_dim, hidden_dim, dropout)
                for _ in range(n_nl_head)
            ]
        )
        self.fc_nl_head.append(
            nn.Sequential(
                nn.Linear(hidden_dim, MAX_MZ),
                nn.Identity(),
            )
        )

        self.fc_gate_head = nn.ModuleList(
            [
                self._make_lin_layer(hidden_dim, hidden_dim, dropout)
                for _ in range(n_gate_head)
            ]
        )
        self.fc_gate_head.append(
            nn.Sequential(nn.Linear(hidden_dim, MAX_MZ), nn.Sigmoid())
        )

    def _make_lin_layer(self, in_weights, hidden, dropout):
        return nn.Sequential(
            nn.Linear(in_weights, hidden), nn.Dropout(dropout), nn.SiLU(inplace=True)
        )

    def forward(self, fingerprints, mol_wts):
        x = self.fc_in(fingerprints)
        y = x
        for i, lin in enumerate(self.fc_middle_layers):
            if i % 2 == 1:
                y = lin(x + y)
            elif i % 2 == 0:
                x = lin(x + y)
        g, y, z = x, x, x
        for _, lin_wt in enumerate(self.fc_gate_head):
            g = lin_wt(g)

        for _, lin_wt in enumerate(self.fc_wt_head):
            y = lin_wt(y)

        for _, lin_nl in enumerate(self.fc_nl_head):
            z = lin_nl(z)
        out = g * y + (1 - g) * nl_to_og_torch(z, mol_wts)
        return nn.functional.relu(out)

    def train_fn(self, optim, loss_fn, train_dl):
        self.train()
        epoch_train_loss = 0
        for mol_fingerprints, mol_weights, spectra in train_dl:
            optim.zero_grad()
            pred_spectra = self.forward(
                mol_fingerprints.to(next(self.parameters()).device, non_blocking=True),
                mol_weights.to(next(self.parameters()).device, non_blocking=True),
            )
            loss = loss_fn(
                pred_spectra,
                spectra.to(next(self.parameters()).device, non_blocking=True),
            )
            loss.backward()
            optim.step()
            epoch_train_loss += loss.detach().cpu()
        return epoch_train_loss / len(train_dl.dataset)

    def eval_fn(self, loss_fn, eval_dl):
        self.eval()
        with torch.no_grad():
            epoch_eval_loss = 0
            for mol_fingerprints, mol_weights, spectra in eval_dl:
                pred_spectra = self.forward(
                    mol_fingerprints.to(
                        next(self.parameters()).device, non_blocking=True
                    ),
                    mol_weights.to(next(self.parameters()).device, non_blocking=True),
                )
                loss = loss_fn(
                    pred_spectra,
                    spectra.to(next(self.parameters()).device, non_blocking=True),
                )
                epoch_eval_loss += loss.detach().cpu()
        return epoch_eval_loss / len(eval_dl.dataset)

    def predict(self, forward_dl):
        self.eval()
        wt_predictions = []
        pbar = tqdm(total=len(forward_dl.dataset))
        with torch.no_grad():
            for fingerprints, molecular_weights in forward_dl:
                predicted_spectra = self.forward(
                    fingerprints.to(next(self.parameters()).device, non_blocking=True),
                    molecular_weights.to(
                        next(self.parameters()).device, non_blocking=True
                    ),
                )
                wt_predictions.append(predicted_spectra)
                pbar.update(predicted_spectra.size(0))
        predictions = torch.cat(wt_predictions, 0).cpu()
        predictions = (
            predictions
            / torch.clamp(torch.max(predictions, dim=1, keepdim=True)[0], min=1e-8)
            * 999
        )
        return predictions.numpy()
