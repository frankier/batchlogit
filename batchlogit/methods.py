import torch
from torch import nn


def lr_one_skl(x: torch.tensor, y: torch.tensor):
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    initial_device = x.device
    x = x.detach()
    model = LogisticRegression()
    with warnings.catch_warnings():

        warnings.simplefilter("ignore", ConvergenceWarning)
        model = model.fit(x, y)
    weight = model.coef_
    bias = model.intercept_
    weight = torch.as_tensor(weight, dtype=torch.float32, device=initial_device)
    bias = torch.as_tensor(bias, dtype=torch.float32, device=initial_device)
    n_iter = model.n_iter_
    return weight, bias, n_iter


def lr_one_cuml(x: torch.tensor, y: torch.tensor):
    from cuml.linear_model import LogisticRegression

    initial_device = x.device
    x = x.detach()
    y = y.float()
    model = LogisticRegression()
    model = model.fit(x, y)
    weight = model.coef_
    bias = model.intercept_
    weight = torch.as_tensor(weight, device=initial_device).float().t()
    bias = torch.as_tensor(bias, device=initial_device).float()
    n_iter = model.solver_model.num_iters
    return weight, bias, n_iter


class StackedRegLogitModel(torch.nn.Module):
    def __init__(self, chunk_size, input_dim, output_dim, C=1):
        super(StackedRegLogitModel, self).__init__()
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, output_dim) for _ in range(chunk_size)]
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
        self.C = C

    def forward(self, x):
        y_all = [linear(vec) for linear, vec in zip(self.linears, x)]
        return torch.stack(y_all, axis=0)

    def forward_loss(self, x, y):
        y_hat_logit = self(x)
        model_loss = self.loss_fn(y_hat_logit[:, :, 0], y)
        # Don't regularize bias to match scikit-learn/cuml
        l2_reg = 0.5 * sum(
            (torch.mm(linear.weight, linear.weight.t()) for linear in self.linears)
        )
        return model_loss + l2_reg / self.C


class RegLogitModel(nn.Module):
    def __init__(self, input_dim, output_dim, C=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
        self.C = C

    def forward(self, x):
        return self.linear(x)

    def forward_loss(self, x, y):
        y_hat_logit = self(x)
        model_loss = self.loss_fn(y_hat_logit[:, 0], y)
        # Don't regularize bias to match scikit-learn/cuml
        l2_reg = 0.5 * torch.mm(self.linear.weight, self.linear.weight.t())
        return model_loss + l2_reg / self.C


def lr_many_pytorch_lbfgs(
    x, y, history_size=10, max_iter=100, max_ls=25, tol=1e-4, C=1,
):
    from torch.optim import LBFGS

    model = StackedRegLogitModel(x.shape[0], x.shape[-1], 1, C=C).to(x.device)
    optimizer = LBFGS(
        model.parameters(),
        lr=1,
        history_size=history_size,
        max_iter=max_iter,
        # XXX: Cannot pass max_ls to strong_wolfe
        line_search_fn="strong_wolfe",
        tolerance_change=0,
        tolerance_grad=tol,
    )

    x_var = x.detach()
    x_var.requires_grad_(True)
    y_var = y.detach().float()

    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        loss = model.forward_loss(x_var, y_var)
        if torch.is_grad_enabled():
            loss.backward()
        return loss

    optimizer.step(closure)
    state = optimizer.state[next(iter(optimizer.state))]
    weights = []
    biases = []
    for linear in model.linears:
        weights.append(linear.weight.detach())
        biases.append(linear.bias.detach())
    return (torch.stack(weights, axis=0), torch.stack(biases, axis=0), state["n_iter"])


def lr_one_pytorch_lbfgs(
    x: torch.tensor,
    y: torch.tensor,
    history_size=10,
    max_iter=100,
    max_ls=25,
    tol=1e-4,
    C=1,
):
    from torch.optim import LBFGS

    model = RegLogitModel(x.shape[-1], 1, C=C).to(x.device)
    optimizer = LBFGS(
        model.parameters(),
        lr=1,
        history_size=history_size,
        max_iter=max_iter,
        # XXX: Cannot pass max_ls to strong_wolfe
        line_search_fn="strong_wolfe",
        tolerance_change=0,
        tolerance_grad=tol,
    )

    x_var = x.detach()
    x_var.requires_grad_(True)
    y_var = y.detach().float()

    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        loss = model.forward_loss(x_var, y_var)
        if torch.is_grad_enabled():
            loss.backward()
        return loss

    optimizer.step(closure)
    state = optimizer.state[next(iter(optimizer.state))]
    return (model.linear.weight.detach(), model.linear.bias.detach(), state["n_iter"])


def lr_one_nlesc_dirac_lbgfs(
    x: torch.tensor,
    y: torch.tensor,
    history_size=10,
    max_iter=100,
    max_ls=25,
    tol=1e-3,
    C=1,
):
    from .vendor.nlesc_dirac_lbgfs import LBFGSNew

    model = RegLogitModel(x.shape[-1], 1, C=C).to(x.device)
    optimizer = LBFGSNew(
        model.parameters(),
        lr=1,
        max_iter=max_iter,
        history_size=history_size,
        tolerance_grad=tol,
        tolerance_change=0,
        line_search_fn=True,
        batch_mode=False,
    )

    x_var = x.detach()
    x_var.requires_grad_(True)
    y_var = y.detach().float()

    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        loss = model.forward_loss(x_var, y_var)
        if torch.is_grad_enabled():
            loss.backward()
        return loss

    optimizer.step(closure)
    state = optimizer.state[next(iter(optimizer.state))]
    return (model.linear.weight.detach(), model.linear.bias.detach(), state["n_iter"])


def lr_one_pytorch_hjmshi_lbfgs(
    x: torch.tensor,
    y: torch.tensor,
    history_size=10,
    max_iter=100,
    max_ls=25,
    tol=1e-4,
    C=1,
):
    print("XXX: This seems to be broken currently.")
    # XXX: Currently broken
    from .vendor.hjmshi_lbfgs import FullBatchLBFGS

    model = RegLogitModel(x.shape[-1], 1, C=C).to(x.device)
    optimizer = FullBatchLBFGS(
        model.parameters(), lr=1, history_size=history_size, line_search="Wolfe",
    )

    x_var = x.detach()
    x_var.requires_grad_(True)
    y_var = y.detach().float()

    def closure():
        loss = model.forward_loss(x_var, y_var)
        return loss

    loss = closure()
    loss.backward()
    n_iter = 1
    while 1:
        options = {
            "closure": closure,
            "current_loss": loss,
            "max_ls": max_ls,
        }
        (
            loss,
            grad,
            lr,
            backtracks,
            clos_evals,
            grad_evals,
            desc_dir,
            fail,
        ) = optimizer.step(options=options)

        grad_max = grad.abs().max()

        if fail:
            raise RuntimeError("Optimizer failure in lr_one_pytorch_hjmshi_lbfgs", fail)
        elif torch.isnan(loss):
            raise RuntimeError("NaN loss in lr_one_pytorch_hjmshi_lbfgs")
        elif grad_max < tol or n_iter == max_iter:
            return (model.linear.weight.detach(), model.linear.bias.detach(), n_iter)
        n_iter += 1
