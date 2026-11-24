import torch
from torch.nn import functional as F

from .storage import StorageBSplines, bsplines_storage


class MultiBSplines:
    def __init__(
        self,
        nodes: torch.Tensor,  # nodes.shape = (nodes_count, postshape)
        degree: int,
        requires_grad=False,
        nodes_completer=None,
        storage: StorageBSplines | None = None,
    ) -> None:
        self.check_init_args(nodes, degree)


        self._nodes_count = nodes.shape[0]
        self._postshape = nodes.shape[1:]  # number of splines
        self._storage = (
            storage
            if storage is not None
            else bsplines_storage(nodes.shape, degree, nodes.dtype, nodes.device)
        )

        self._nodes_completer = (
            nodes_completer
            if nodes_completer is not None
            else MultiBSplines._complete_nodes_shift
        )

        self._degree = degree
        self._nodes = self._nodes_completer(nodes, degree)
        self._basis_size = MultiBSplines.get_basis_size(nodes.shape[0], degree)
        interval_count = self._nodes_count - 1
        self._interval_basis_indices = (
            torch.arange(interval_count, device=nodes.device)[:, None]
            + torch.arange(degree + 1, device=nodes.device)[None, :]
        )

        # Polynomial coefficients are stored per interval in the normalized
        # local coordinate u = (x - left) / (right - left).  Keeping u in
        # [0, 1] avoids the cancellation caused by powers of the global x.
        self._repr_requires_grad = requires_grad
        self._repr = self._create_repr(requires_grad)

        nodes_perm = [i for i in range(1, self._nodes.dim())] + [0]

        self._def_nodes = self._nodes[
            self._degree : self._nodes_count + self._degree, ...
        ].permute(*nodes_perm).contiguous()

    @staticmethod
    def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        """Вычисляет `numerator / denominator`, подставляя 1 вместо нулевого знаменателя.

        Нулевой знаменатель соответствует кратному узлу. У базисной функции
        на этом интервале нулевая ширина носителя и нулевое значение. Любая
        конечная подстановка сохраняет результат и позволяет избежать NaN.
        """
        safe_denominator = torch.where(
            denominator != 0, denominator, torch.ones_like(denominator)
        )
        return numerator / safe_denominator

    def _affine_recursion_coefficients(
        self,
        new_degree: int,
        interval_count: int,
        interval_left: torch.Tensor,
        interval_width: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Вычисляет `alpha_i(u)` и `beta_{i+1}(u)` для рекурсии Кокса — де Бура.

            B_{i,p}(u) = alpha_i(u) B_{i,p-1}(u) + beta_{i+1}(u) B_{i+1,p-1}(u)

        На каждом интервале активны `new_degree + 1` базисных функций степени
        `new_degree`. Для каждой возвращается пара коэффициентов при постоянном
        и линейном членах по локальной координате `u`.

            alpha_i(x) = (x - t_i) / (t_{i+p} - t_i)
            beta_{i+1}(x) = (t_{i+p+1} - x) / (t_{i+p+1} - t_{i+1})
        """
        first_interval = self._degree
        alpha_constant, alpha_linear = [], []
        beta_constant, beta_linear = [], []
        for active_index in range(new_degree + 1):
            basis_start = first_interval - new_degree + active_index
            basis_left = self._nodes[basis_start : basis_start + interval_count]
            basis_right = self._nodes[
                basis_start + new_degree + 1
                : basis_start + new_degree + interval_count + 1
            ]
            left_span = (
                self._nodes[
                    basis_start + new_degree : basis_start + new_degree + interval_count
                ]
                - basis_left
            )
            right_span = basis_right - self._nodes[
                basis_start + 1 : basis_start + interval_count + 1
            ]

            alpha_constant.append(self._safe_ratio(interval_left - basis_left, left_span))
            alpha_linear.append(self._safe_ratio(interval_width, left_span))
            beta_constant.append(self._safe_ratio(basis_right - interval_left, right_span))
            beta_linear.append(self._safe_ratio(-interval_width, right_span))

        return (
            torch.stack(alpha_constant),
            torch.stack(alpha_linear),
            torch.stack(beta_constant),
            torch.stack(beta_linear),
        )

    def _create_repr(self, requires_grad):
        """Строит коэффициенты в степенном базисе для активных сплайнов на каждом интервале.

        Коэффициенты сплайнов степени `p` вычисляются рекурсией Кокса — де Бура
        в локальной координате `u = (x - left) / (right - left)` (см.
        `_affine_recursion_coefficients`). Степени `u` лежат в `[0, 1]`, что
        позволяет избежать потери точности при вычислениях со степенями глобальной
        координаты `x`.

        В `degree_coefficients[power, active_index, interval, *postshape]` хранится
        коэффициент при `u**power` для активной функции с индексом `active_index`
        на каждом интервале. С ростом степени добавляется по одной активной
        функции и одному степенному члену.
        """
        with torch.no_grad():
            interval_count = self._nodes_count - 1

            # On one interval, a degree-p spline basis has only p + 1 active
            # functions. Build exactly those local polynomials instead of the
            # previous dense basis-by-interval table.
            
            # (power, active_count, interval_count, postshape)
            degree_coefficients = torch.ones(
                (1, 1, interval_count, *self._postshape),
                dtype=self._nodes.dtype,
                device=self._nodes.device,
            )

            first_interval = self._degree
            interval_left = self._nodes[
                first_interval : first_interval + interval_count
            ]
            interval_width = (
                self._nodes[first_interval + 1 : first_interval + interval_count + 1]
                - interval_left
            )

            for new_degree in range(1, self._degree + 1):
                alpha_constant, alpha_linear, beta_constant, beta_linear = (
                    self._affine_recursion_coefficients(
                        new_degree, interval_count, interval_left, interval_width
                    )
                )

                # Active function j of the new degree is alpha_j times the
                # PREVIOUS degree's function at slot j - 1, plus beta_j times
                # its function at slot j (unchanged). E.g. going from degree
                # 0 (one active function `b0`) to degree 1 (two, j=0 and
                # j=1): slot 0 is pure beta_0 * b0 (its alpha would need the
                # nonexistent slot -1), slot 1 is pure alpha_1 * b0 (its beta
                # would need the nonexistent slot 1). Padding zero on the
                # side that would otherwise go out of range realizes that:
                # `shifted_for_alpha[j]` reads `degree_coefficients[j - 1]`,
                # `shifted_for_beta[j]` reads `degree_coefficients[j]`, and
                # both fall back to 0 exactly at the missing ends.
                zero = degree_coefficients.new_zeros(
                    (degree_coefficients.shape[0], 1, *degree_coefficients.shape[2:])
                )
                shifted_for_alpha = torch.cat((zero, degree_coefficients), dim=1)
                shifted_for_beta = torch.cat((degree_coefficients, zero), dim=1)

                next_coefficients = torch.zeros(
                    (
                        new_degree + 1, # poly degree
                        new_degree + 1, # count of active spline
                        interval_count,
                        *self._postshape,
                    ),
                    dtype=self._nodes.dtype,
                    device=self._nodes.device,
                )
                # Constant terms land at the same power; each alpha/beta's
                # linear-in-u term raises the power of what it multiplies by
                # one, hence the `[1:] +=` at power index one higher.
                next_coefficients[:-1] = (
                    alpha_constant.unsqueeze(0) * shifted_for_alpha
                    + beta_constant.unsqueeze(0) * shifted_for_beta
                )
                next_coefficients[1:] += (
                    alpha_linear.unsqueeze(0) * shifted_for_alpha
                    + beta_linear.unsqueeze(0) * shifted_for_beta
                )
                degree_coefficients = next_coefficients

            perm_dims = [0] + [i for i in range(3, degree_coefficients.dim())] + [1, 2]
            compact = degree_coefficients.permute(*perm_dims)
            representation = self._storage.local_coefficients
            representation.zero_()
            representation[..., 1:-1].copy_(compact)
            return representation.requires_grad_(requires_grad)

    def __call__(
        self,
        x: torch.Tensor,
        coef: torch.Tensor,
        bvalues: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        is_sorted: bool = False,
    ) -> torch.Tensor:

        # coef.shape = (postshape, basis_size)
        if bvalues is not None:
            extra_count = coef.ndim - len(self._postshape) - 1
            expanded_basis = bvalues.reshape(
                *((1,) * extra_count), *bvalues.shape
            )
            return (expanded_basis * coef[..., None]).sum(dim=-2)

        selected, expanded_local_x, _, overshoot = self._selected_polynomial(x, coef)
        value, local_derivative = self._horner(selected, expanded_local_x)
        return value + local_derivative * overshoot

    def value_and_derivative(
        self,
        x: torch.Tensor,
        coef: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Вычисляет значения сплайнов и их производные по ``x``."""
        selected, expanded_local_x, expanded_width, overshoot = self._selected_polynomial(
            x, coef
        )
        value, local_derivative = self._horner(selected, expanded_local_x)
        return value + local_derivative * overshoot, local_derivative / expanded_width

    @staticmethod
    def _horner(
        selected: torch.Tensor, local_x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = selected[-1]
        local_derivative = torch.zeros_like(value)
        for power in range(len(selected) - 2, -1, -1):
            local_derivative = local_derivative * local_x + value
            value = value * local_x + selected[power]
        return value, local_derivative

    def _selected_polynomial(
        self,
        x: torch.Tensor,
        coef: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        intervals, local_x, cell_indices, overshoot = self._locate(x)
        # Combine trainable basis weights with the local basis polynomials once
        # per knot interval.  Selecting the resulting spline polynomial per
        # point avoids tensors shaped (..., basis_size, point_count) and
        # (..., degree + 1, point_count) in the coefficient path.
        expected_postshape = coef.shape[
            coef.ndim - len(self._postshape) - 1 : -1
        ]
        if expected_postshape != self._postshape or coef.shape[-1] != self._basis_size:
            raise ValueError(
                "B-spline coefficients must end with shape "
                f"{self._postshape + (self._basis_size,)}, got {tuple(coef.shape)}"
            )
        extra_count = coef.ndim - len(self._postshape) - 1
        interval_coef = coef[..., self._interval_basis_indices].movedim(-1, -2)
        representation = self._repr.reshape(
            self._degree + 1,
            *((1,) * extra_count),
            *self._repr.shape[1:],
        )
        polynomial = (
            representation[..., 1:-1] * interval_coef.unsqueeze(0)
        ).sum(dim=-2)
        polynomial = F.pad(polynomial, (1, 1), value=0)
        expanded_intervals = intervals.reshape(
            *((1,) * extra_count), *intervals.shape
        ).expand(*coef.shape[:extra_count], *intervals.shape)
        gather_indices = expanded_intervals.unsqueeze(0).expand(
            self._degree + 1, *expanded_intervals.shape
        )
        selected = torch.gather(polynomial, -1, gather_indices)
        expanded_local_x = local_x.reshape(
            *((1,) * extra_count), *local_x.shape
        )
        expanded_overshoot = overshoot.reshape(
            *((1,) * extra_count), *overshoot.shape
        )
        interval_widths = self._def_nodes[..., 1:] - self._def_nodes[..., :-1]
        selected_widths = torch.gather(interval_widths, -1, cell_indices)
        expanded_widths = selected_widths.reshape(
            *((1,) * extra_count), *selected_widths.shape
        )
        return selected, expanded_local_x, expanded_widths, expanded_overshoot

    # get bsplines values of bspline on x
    # x.shape = (postshape, N)
    def _locate(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.shape[:-1] != self._postshape:
            raise ValueError(
                "B-spline input must have shape "
                f"{self._postshape + (x.shape[-1],)}, got {tuple(x.shape)}"
            )
        if x.dtype != self._nodes.dtype or x.device != self._nodes.device:
            raise ValueError(
                "B-spline input must have the same dtype and device as the grid"
            )

        # Layer inputs are commonly produced with expand(), while searchsorted
        # otherwise performs the same copy implicitly and emits a warning.
        x = x.contiguous()
        raw_intervals = torch.searchsorted(self._def_nodes, x, right=True)
        cell_indices = (raw_intervals - 1).clamp(0, self._nodes_count - 2)
        interval_left = torch.gather(
            self._def_nodes[..., :-1], -1, cell_indices
        )
        interval_right = torch.gather(
            self._def_nodes[..., 1:], -1, cell_indices
        )
        raw_local_x = (x - interval_left) / (interval_right - interval_left)
        local_x = raw_local_x.clamp(0.0, 1.0)
        # `cell_indices + 1` is the index into the padded per-interval
        # polynomial tables (`_repr`, and `_selected_polynomial`'s local
        # `polynomial`) that matches `cell_indices`/`local_x` above: the last
        # interval's polynomial, evaluated at its right end, for any x at or
        # beyond the right boundary node. The raw `searchsorted(..., right=True)` result
        # instead puts x == nodes[-1] one slot past the real data, selecting
        # the table's zero pad there -- which used to make the spline
        # discontinuously drop to zero exactly at (and beyond) the right
        # boundary node, while the left boundary (already at cell 0) stayed
        # continuous.
        return cell_indices + 1, local_x, cell_indices, raw_local_x - local_x

    def _get_active_bvalues(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        intervals, local_x, cell_indices, overshoot = self._locate(x)
        intervals = intervals.unsqueeze(0).unsqueeze(-2).expand(*self._repr.shape[:-1], -1)

        selected = torch.gather(
            self._repr, -1, intervals
        )  # (degree, postshape, active_count, N)

        value, local_derivative = self._horner(selected, local_x[..., None, :])
        result = value + local_derivative * overshoot[..., None, :]

        offset_shape = (
            (1,) * len(self._postshape) + (self._degree + 1, 1)
        )
        active_offsets = torch.arange(
            self._degree + 1,
            dtype=cell_indices.dtype,
            device=cell_indices.device,
        ).view(offset_shape)
        active_indices = cell_indices.unsqueeze(-2) + active_offsets
        return result, active_indices

    def get_bvalues(self, x: torch.Tensor) -> torch.Tensor:
        active_values, active_indices = self._get_active_bvalues(x)
        dense = active_values.new_zeros(
            (*self._postshape, self._basis_size, x.shape[-1])
        )
        return dense.scatter(-2, active_indices, active_values)

    @property
    def basis_size(self):
        return self._basis_size

    @property
    def repr(self):
        if self._repr is None:
            self._repr = self._create_repr(self._repr_requires_grad)
        return self._repr

    @property
    def degree(self):
        return self._degree

    @property
    def nodes(self):
        return self._nodes

    @staticmethod
    def _complete_nodes_shift(
        nodes: torch.Tensor,
        degree: int,
        out=None,
    ):
        nodes_count = nodes.shape[0]
        interval_count = torch.Size((nodes_count + 2 * degree,))
        completed = (
            out
            if out is not None
            else torch.empty(
                interval_count + nodes.shape[1:],
                dtype=nodes.dtype,
                device=nodes.device,
            )
        )
        dtype = completed.dtype
        device = completed.device

        scale_shape = (-1, *([1] * (nodes.dim() - 1)))

        shift = (nodes[[-1], ...] - nodes[[0], ...]) / (nodes_count - 1)
        scale = torch.linspace(-1, 0, degree + 1, dtype=dtype, device=device).view(
            *scale_shape
        )
        completed[0 : degree + 1, ...] = nodes[[0], ...] + (shift * degree * scale)

        completed[degree + 1 : degree + nodes_count - 1, ...] = nodes[
            1 : nodes_count - 1, ...
        ]

        scale = torch.linspace(0, 1, degree + 1, dtype=dtype, device=device).view(
            *scale_shape
        )
        completed[degree + nodes_count - 1 :, ...] = nodes[[-1], ...] + (
            shift * degree * scale
        )
        return completed

    @staticmethod
    def get_basis_size(nodes_count, degree):
        return nodes_count + degree - 1

    @staticmethod
    def check_init_args(nodes: torch.Tensor, degree: int) -> None:
        if degree < 0:
            raise ValueError("B-spline degree must be non-negative")
        if nodes.ndim == 0 or nodes.shape[0] < 2:
            raise ValueError("B-spline grid must contain at least two nodes")
        if not torch.isfinite(nodes).all():
            raise ValueError("B-spline grid contains non-finite nodes")
        if (nodes.diff(dim=0) <= 0).any():
            raise ValueError("B-spline grid nodes must be strictly increasing")
