import logging
from enum import Enum, auto

import torch
from torch import nn
from torch.nn import functional as F

from .bsplines import MultiBSplines
from .initialization import (
    KANInit,
    glorot_std,
    lecun_std,
    power_law_std,
    uniform_samples,
    uniform_variance,
    value_and_derivative,
)
from .storage import bsplines_storage

log = logging.getLogger(__name__)

class KANType(Enum):
    BSPLINE = auto()
    CHEBY = auto()



class BSplinesNN(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        intervals_count,
        degree,
        bias_func=None,
        initial_boundary: torch.Tensor | None = None,
        grid_update_regularization: float = 1e-5,
        grid_update_max_relative_error: float = 2e-2,
        grid_update_max_pointwise_error: float = 5e-2,
        init: KANInit | None = None,
        dtype=torch.float64,
        device=None,
    ) -> None:
        super().__init__()

        self._dtype = dtype
        self._device = torch.device("cpu") if device is None else torch.device(device)

        self._uniform_eps = 0.1

        self._in_features = in_features
        self._out_features = out_features
        self._intervals_count = intervals_count
        self._degree = degree
        if grid_update_regularization <= 0:
            raise ValueError("grid_update_regularization must be positive")
        self._grid_update_regularization = grid_update_regularization
        if grid_update_max_relative_error <= 0:
            raise ValueError("grid_update_max_relative_error must be positive")
        self._grid_update_max_relative_error = grid_update_max_relative_error
        if grid_update_max_pointwise_error <= 0:
            raise ValueError("grid_update_max_pointwise_error must be positive")
        self._grid_update_max_pointwise_error = grid_update_max_pointwise_error

        self._basis_size = MultiBSplines.get_basis_size(intervals_count + 1, degree)

        self._bias_func = bias_func if bias_func is not None else nn.Tanh()
        self._bias = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                dtype=dtype,
                device=self._device,
            )
        )

        self._coef = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                self._basis_size,
                dtype=dtype,
                device=self._device,
            )
        )

        # A trainable per-edge multiplier on the spline term, matching pykan's
        # `scale_sp` (`KANLayer.__init__`): without it, the spline's own
        # magnitude is fixed by its coefficients alone, with no separate knob
        # to grow or shrink its contribution relative to the SiLU term
        # (`_bias`) the way pykan's edges can. `1/sqrt(in_features)` mirrors
        # pykan's deterministic default (`scale_sp=1.0` there).
        self._splineScale = nn.Parameter(
            torch.empty(
                (out_features, in_features),
                dtype=dtype,
                device=self._device,
            )
        )

        boundary = (
            initial_boundary.to(dtype=dtype, device=self._device)
            if initial_boundary is not None
            else torch.tensor([-1.0, 1.0], dtype=dtype, device=self._device)[
                :, None
            ].expand((2, in_features))
        )
        if boundary.ndim == 3:
            if not torch.equal(boundary, boundary[:, :1].expand_as(boundary)):
                raise ValueError("KAN output-specific grid boundaries are unsupported")
            boundary = boundary[:, 0]
        # shape = (intevals_count, postshape)
        grid = self.grid_from_boundary(boundary[0], boundary[1])
        self.register_buffer("_grid", grid)
        self._rebuild_bspline()
        self._initialize_parameters(init if init is not None else KANInit(), boundary)

        self._need_update = False
        self._last_grid_update_status = "never"
        self.register_buffer(
            "_last_rejected_grid",
            torch.empty(0, dtype=dtype, device=self._device),
            persistent=False,
        )
        self._last_rejected_coef_version = -1

    def _rebuild_bspline(self) -> None:
        self._dtype = self._grid.dtype
        self._device = self._grid.device
        self._storage = bsplines_storage(
            self._grid.shape,
            self._degree,
            dtype=self._dtype,
            device=self._device,
        )
        self._bspline = MultiBSplines(
            self._grid,
            self._degree,
            storage=self._storage,
        )

    def _initialize_parameters(self, init: KANInit, boundary: torch.Tensor) -> None:
        with torch.no_grad():
            if init.scheme == "default":
                nn.init.xavier_uniform_(self._bias)
                self._coef.normal_(mean=0.0, std=0.01)
                self._splineScale.fill_(1.0 / self._in_features**0.5)
                return
            self._splineScale.fill_(1.0)
            if init.scheme == "baseline":
                nn.init.xavier_uniform_(self._bias)
                self._coef.normal_(mean=0.0, std=0.1)
                return
            residual_std, spline_std = self._init_stds(init, boundary)
            self._bias.copy_(torch.randn_like(self._bias) * residual_std)
            self._coef.copy_(torch.randn_like(self._coef) * spline_std[:, None])

    def _init_stds(
        self, init: KANInit, boundary: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        components = self._basis_size + 1
        fan_in, fan_out = self._in_features, self._out_features
        if init.scheme == "power_law":
            return tuple(
                torch.full(
                    (fan_in,),
                    power_law_std(fan_in, components, exponent),
                    dtype=self._dtype,
                    device=self._device,
                )
                for exponent in (init.alpha, init.beta)
            )

        x = uniform_samples(boundary[0], boundary[1])
        residual, residual_derivative = value_and_derivative(self._bias_func, x)
        identity = torch.eye(self._basis_size, dtype=self._dtype, device=self._device)
        basis_coef = identity[:, None, :].expand(-1, fan_in, -1)
        basis = self._bspline.get_bvalues(x.T)
        _, basis_derivative = self._bspline.value_and_derivative(x.T, basis_coef)

        residual_moment = residual.square().mean(dim=0)
        basis_moment = basis.square().mean(dim=(-2, -1))
        if init.scheme == "lecun":
            variance = uniform_variance(boundary[0], boundary[1])
            return (
                lecun_std(variance, fan_in, components, residual_moment),
                lecun_std(variance, fan_in, components, basis_moment),
            )
        return (
            glorot_std(
                fan_in,
                fan_out,
                components,
                residual_moment,
                residual_derivative.square().mean(dim=0),
            ),
            glorot_std(
                fan_in,
                fan_out,
                components,
                basis_moment,
                basis_derivative.square().mean(dim=(0, -1)),
            ),
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        grid_key = prefix + "_grid"
        loaded_grid = state_dict.get(grid_key)
        if (
            loaded_grid is not None
            and loaded_grid.ndim == 3
            and self._grid.ndim == 2
        ):
            shared_grid = loaded_grid[:, :1].expand_as(loaded_grid)
            if not torch.equal(loaded_grid, shared_grid):
                error_msgs.append(
                    f"{grid_key} contains output-specific grids and cannot be "
                    "loaded into a shared-input-grid KAN layer"
                )
            else:
                state_dict[grid_key] = loaded_grid[:, 0]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._rebuild_bspline()

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._rebuild_bspline()
        self._last_rejected_grid = self._last_rejected_grid.new_empty(0)
        self._last_rejected_coef_version = -1
        return result

    # x.shape = (batch_size, in_features)
    # output.shape = (batch_size, out_features)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temp = F.linear(self._bias_func(x), self._bias)
        spline_input = x.movedim(0, 1)
        raw_spline = self._bspline(spline_input, self._coef).movedim(-1, 0)
        mid = (self._splineScale * raw_spline).sum(-1)
        return temp + mid

    def forward_with_jacobian(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Возвращает значения слоя и якобиан ``dy/dx`` для каждого элемента выборки."""
        activation, activation_derivative = torch.func.jvp(
            self._bias_func, (x,), (torch.ones_like(x),)
        )
        spline_value, spline_derivative = self._bspline.value_and_derivative(
            x.movedim(0, 1), self._coef
        )
        spline_value = self._splineScale * spline_value.movedim(-1, 0)
        spline_derivative = self._splineScale * spline_derivative.movedim(-1, 0)
        values = F.linear(activation, self._bias) + spline_value.sum(dim=-1)
        local_jacobian = (
            self._bias.unsqueeze(0) * activation_derivative.unsqueeze(1)
            + spline_derivative
        )
        return values, local_jacobian

    # x.shape = (batch_size, in_features)
    def update_grid(
        self,
        x: torch.Tensor,
        *,
        debug_label: str | None = None,
        debug: bool = True,
    ) -> bool:
        with torch.no_grad():
            if x.ndim != 2 or x.shape[1] != self._in_features:
                raise ValueError(
                    "KAN grid update expects input with shape "
                    f"(batch_size, {self._in_features})"
                )
            if x.shape[0] == 0:
                raise ValueError("cannot update KAN grid from an empty batch")
            if not torch.isfinite(x).all():
                raise ValueError("cannot update KAN grid from non-finite values")
            sort = torch.sort(x, dim=0)[0]
            new_grid = self.create_grid(sort, True)
            grid_diff = new_grid.diff(dim=0)
            if not torch.isfinite(new_grid).all() or (grid_diff <= 0).any():
                raise RuntimeError("KAN grid update produced an invalid grid")
            grid_scale = (self._grid[-1] - self._grid[0]).abs().max().clamp_min(
                torch.finfo(self._grid.dtype).eps
            )
            relative_grid_change = (
                (new_grid - self._grid).abs().max() / grid_scale
            )
            unchanged_tolerance = 128 * torch.finfo(self._grid.dtype).eps
            if relative_grid_change <= unchanged_tolerance:
                self._last_grid_update_status = "unchanged"
                if debug:
                    layer_info = (
                        f" layer={debug_label}" if debug_label is not None else ""
                    )
                    log.info(
                        f"[KAN][grid]{layer_info} status=unchanged "
                        f"relative_grid_change={relative_grid_change.item():.6e}"
                    )
                return False
            if (
                self._last_rejected_coef_version == self._coef._version
                and torch.equal(self._last_rejected_grid, new_grid)
            ):
                self._last_grid_update_status = "rejected_cached"
                if debug:
                    layer_info = (
                        f" layer={debug_label}" if debug_label is not None else ""
                    )
                    log.info(f"[KAN][grid]{layer_info} status=rejected_cached")
                return False

            new_storage = bsplines_storage(
                new_grid.shape,
                self._degree,
                dtype=new_grid.dtype,
                device=new_grid.device,
            )
            new_spline = MultiBSplines(
                new_grid,
                self._degree,
                storage=new_storage,
            )

            # The input points are suitable for choosing knot quantiles, but not
            # for transferring coefficients: repeated values can make a large
            # matrix have very low rank.  Sample every interval of the new grid
            # so the least-squares system is overdetermined and covers all basis
            # functions.
            transfer_count = max(4 * self._basis_size, 64)
            transfer_points = self._grid_interval_points(
                new_grid, transfer_count, offset=0.5
            )
            new_bvalues = new_spline.get_bvalues(transfer_points)
            curr_values = self._bspline(transfer_points, self._coef)
            least_squares_matrix = new_bvalues.transpose(-1, -2)
            least_squares_matrix = least_squares_matrix.unsqueeze(0).expand(
                self._out_features, *least_squares_matrix.shape
            )
            new_coef = self._solve_lstsq(
                least_squares_matrix,
                curr_values,
            )
            if not torch.isfinite(new_coef).all():
                raise RuntimeError("KAN grid update produced non-finite coefficients")

            # Validate between the fitting points before mutating the layer. A
            # grid change also changes the spline space, so exact preservation is
            # not always possible; a large change is safer to reject.
            validation_points = self._grid_interval_points(
                new_grid, 2 * transfer_count, offset=0.25
            )
            values_before = self._bspline(validation_points, self._coef)
            values_after = new_spline(validation_points, new_coef)
            difference = values_after - values_before
            denominator = torch.linalg.vector_norm(values_before).clamp_min(
                torch.finfo(values_before.dtype).eps
            )
            relative_error = torch.linalg.vector_norm(difference) / denominator
            value_scale = values_before.abs().max().clamp_min(
                torch.finfo(values_before.dtype).eps
            )
            max_relative_error = difference.abs().max() / value_scale
            rejected = (
                not torch.isfinite(relative_error)
                or relative_error > self._grid_update_max_relative_error
                or max_relative_error > self._grid_update_max_pointwise_error
            )
            layer_info = f" layer={debug_label}" if debug_label is not None else ""
            if debug:
                log.info(
                    f"[KAN][grid]{layer_info} "
                    f"status={'rejected' if rejected else 'accepted'} "
                    f"equations={transfer_points.shape[-1]} "
                    f"coefficients={self._basis_size} "
                    f"relative_l2={relative_error.item():.6e} "
                    f"max_abs={difference.abs().max().item():.6e} "
                    f"mean_abs={difference.abs().mean().item():.6e}"
                )
            if rejected:
                self._last_grid_update_status = "rejected"
                self._last_rejected_grid = new_grid.clone()
                self._last_rejected_coef_version = self._coef._version
                return False

            self._coef.copy_(new_coef)
            self._storage = new_storage
            self._bspline = new_spline
            self._grid.copy_(new_grid)
            self._last_grid_update_status = "accepted"
            self._last_rejected_grid = self._last_rejected_grid.new_empty(0)
            self._last_rejected_coef_version = -1
            return True

    @staticmethod
    def _grid_interval_points(
        grid: torch.Tensor, count: int, *, offset: float
    ) -> torch.Tensor:
        if not 0 < offset < 1:
            raise ValueError("interval sample offset must be between zero and one")
        interval_count = grid.shape[0] - 1
        samples_per_interval = max(1, (count + interval_count - 1) // interval_count)
        alpha = (
            torch.arange(
                samples_per_interval,
                dtype=grid.dtype,
                device=grid.device,
            )
            + offset
        ) / samples_per_interval
        left = grid[:-1].movedim(0, -1)
        width = grid.diff(dim=0).movedim(0, -1)
        return (
            left[..., :, None] + width[..., :, None] * alpha
        ).flatten(start_dim=-2)

    def _regularized_lstsq(
        self,
        matrix: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Решает пакет задач МНК масштабированным фильтром Тихонова на основе SVD."""
        if matrix.shape[-2] != target.shape[-1]:
            raise ValueError("least-squares matrix and target sizes do not match")
        left, singular, right_h = torch.linalg.svd(matrix, full_matrices=False)
        largest = singular[..., 0]
        cutoff = (
            largest
            * max(matrix.shape[-2:])
            * torch.finfo(singular.dtype).eps
        ).clamp_min(torch.finfo(singular.dtype).eps)
        condition = largest / singular[..., -1].clamp_min(cutoff)
        underdetermined = matrix.shape[-2] < matrix.shape[-1]
        ill_conditioned = (
            (singular[..., -1] <= cutoff)
            | (condition > 1e4)
            | underdetermined
        )
        scale = largest.square().clamp_min(torch.finfo(singular.dtype).eps)
        regularization = torch.where(
            ill_conditioned,
            self._grid_update_regularization * scale,
            torch.zeros_like(scale),
        )
        factors = singular / (
            singular.square()
            + regularization[..., None]
        )
        factors = torch.where(singular > cutoff[..., None], factors, 0)
        projected = torch.einsum("...nb,...n->...b", left, target)
        return torch.einsum("...bk,...b->...k", right_h, factors * projected)

    def _solve_lstsq(
        self,
        matrix: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Решает задачу МНК быстрым QR-разложением для полного ранга и использует SVD при сбое."""
        try:
            solution = torch.linalg.lstsq(
                matrix,
                target[..., None],
                driver="gels",
            ).solution[..., 0]
        except RuntimeError:
            return self._regularized_lstsq(matrix, target)
        if not torch.isfinite(solution).all():
            return self._regularized_lstsq(matrix, target)
        return solution

    # min ||AX - B||
    # B = curr_values, curr_values.shape = (out_features, in_features, batch_size)
    # X: X.shape = (out_features, in_features, basis_size), (out_features, in_features) = some_batch
    # A: A.shape = (out_features, in_features, batch_size, basis_size)

    # x.shape = (batch_size, in_features)
    def create_grid(self, x: torch.Tensor, is_sorted: bool) -> torch.Tensor:
        uniform = self.create_uniform_grid(x, is_sorted)
        adaptive = self.create_adaptive_grid(x, is_sorted)
        if uniform.shape != adaptive.shape:
            raise RuntimeError("uniform and adaptive KAN grids have different shapes")

        new_grid = self._uniform_eps * uniform + (1 - self._uniform_eps) * adaptive
        diff = new_grid.diff(dim=0)
        invalid = (diff <= torch.finfo(new_grid.dtype).eps).any(dim=0)
        if invalid.any():
            new_grid[:, invalid] = self._grid[:, invalid]
        return new_grid

    # x.shape = (batch_size, out_features, in_features)
    def create_adaptive_grid(self, x: torch.Tensor, is_sorted: bool) -> torch.Tensor:
        batch_size = x.shape[0]
        sort = x if is_sorted else torch.sort(x, dim=0)[0]
        grid = torch.empty(
            (self._intervals_count + 1, *sort.shape[1:]),
            dtype=sort.dtype,
            device=sort.device,
        )
        ids = torch.arange(self._intervals_count, device=sort.device)
        ids = torch.div(
            ids * batch_size,
            self._intervals_count,
            rounding_mode="floor",
        )
        grid[:-1] = sort[ids]
        grid[-1] = sort[-1]
        margin = (sort[-1] - sort[0]) / self._intervals_count
        grid[0] = sort[0] - margin
        grid[-1] = sort[-1] + margin

        return grid

    # x.shape = (batch_size, out_features, in_features)
    def create_uniform_grid(self, x: torch.Tensor, is_sorted: bool) -> torch.Tensor:
        start, end = (x[0], x[-1]) if is_sorted else torch.aminmax(x, dim=0)
        temp = self.grid_from_boundary(start, end)

        return temp

    # start.shape = end.shape = postshape
    # out.shape = (intervals_count + 1, postshape)
    def grid_from_boundary(self, start: torch.Tensor, end: torch.Tensor):
        margin = (end - start) / self._intervals_count
        start = start - margin
        end = end + margin
        temp = torch.linspace(
            0.0, 1.0, self._intervals_count + 1, dtype=self._dtype, device=self._device
        )
        temp = start[..., None] + (end - start)[..., None] * temp
        return temp.movedim(-1, 0).contiguous()

    @property
    def coefficient_parameter(self) -> nn.Parameter:
        return self._coef

    @property
    def last_grid_update_status(self) -> str:
        return self._last_grid_update_status
