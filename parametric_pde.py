from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import spsolve

from fem import (
    generate_fe_mesh,
    precompute_p1,
    assemble_load,
    assemble_stiffness,
    evaluate_fe_uniform,
    integrate_fe,
)


# ============================================================
# Solution classes
# ============================================================

class Solution(ABC):
    """Scalar function u : [0, 1]^2 -> R."""

    @abstractmethod
    def evaluate(self, x: ArrayLike) -> NDArray:
        """Evaluate the solution at one or more points."""
        pass

    def __call__(self, x: ArrayLike) -> NDArray:
        return self.evaluate(x)

    @abstractmethod
    def integrate(self) -> float:
        """Integrate the solution over [0, 1]^2."""
        pass


@dataclass(frozen=True)
class AnalyticalSolution(Solution):
    """Analytical solution with optional exact integral."""

    function: Callable[[NDArray], NDArray]
    exact_integral: float | None = None

    def evaluate(self, x: ArrayLike) -> NDArray:
        x = np.atleast_2d(
            np.asarray(x, dtype=float)
        )
        return np.asarray(
            self.function(x),
            dtype=float,
        )

    def integrate(self) -> float:
        if self.exact_integral is None:
            raise ValueError(
                "No exact integral was provided."
            )

        return float(self.exact_integral)


@dataclass(frozen=True)
class FEMSolution(Solution):
    """P1 finite-element solution on a uniform triangular mesh."""

    coefficients: NDArray
    nodes: NDArray
    elements: NDArray
    areas: NDArray
    n: int
    computed_integral: float | None = None

    def evaluate(self, x: ArrayLike) -> NDArray:
        return evaluate_fe_uniform(
            x,
            self.coefficients,
            self.n,
        )

    def integrate(self) -> float:
        if self.computed_integral is not None:
            return float(self.computed_integral)

        return float(
            integrate_fe(
                self.elements,
                self.areas,
                self.coefficients,
            )
        )


# ============================================================
# Parametric PDE base class
# ============================================================

class ParametricPDE(ABC):
    """
    Base class for parametric PDEs.

    Subclasses specify the actual PDE and how it is solved.
    """

    def __init__(
        self,
        dim: int = 20,
        decay: float = 1.25,
        baseline: float = 1.0,
        boundary_data: float = 0.0,
    ):
        self.s = int(dim)
        self.decay = float(decay)
        self.baseline = float(baseline)
        self.boundary_data = float(boundary_data)

    def set_boundary_data(
        self,
        boundary_data: float,
    ) -> None:
        """Change the constant Dirichlet boundary condition."""
        self.boundary_data = float(boundary_data)

    def _validate_parameter(
        self,
        y: ArrayLike,
    ) -> NDArray:
        """Convert y to an array and check its dimension."""
        y = np.asarray(y, dtype=float)

        if y.shape != (self.s,):
            raise ValueError(
                f"Expected y with shape ({self.s},), "
                f"got {y.shape}."
            )

        return y

    @abstractmethod
    def solve(
        self,
        y: ArrayLike,
    ) -> Solution:
        """Solve the PDE for parameter vector y."""
        pass


# ============================================================
# Finite-element base class
# ============================================================

class FiniteElementParametricPDE(ParametricPDE):
    """
    Base class for P1 finite-element problems of the form

        -div(a(x, y) grad u) = f

    on [0, 1]^2 with constant Dirichlet boundary data.

    Subclasses only need to specify the diffusion coefficient
    a(x, y).
    """

    def __init__(
        self,
        dim: int = 20,
        decay: float = 1.25,
        baseline: float = 1.0,
        boundary_data: float = 0.0,
        forcing: Callable[[NDArray], NDArray] | None = None,
        level: int = 5,
    ):
        super().__init__(
            dim=dim,
            decay=decay,
            baseline=baseline,
            boundary_data=boundary_data,
        )

        self.level = int(level)

        # Mesh
        (
            self.nodes,
            self.elements,
            self.interior,
            self.n,
        ) = generate_fe_mesh(self.level)

        self.boundary = np.setdiff1d(
            np.arange(len(self.nodes)),
            self.interior,
        )

        # Quantities depending only on the mesh
        (
            self.areas,
            self.local_stiffness,
            self.rows,
            self.cols,
            self.qpoints,
        ) = precompute_p1(
            self.nodes,
            self.elements,
        )

        self.forcing = (
            forcing
            if forcing is not None
            else self._default_forcing
        )

        # The load vector is independent of y.
        self.rhs = assemble_load(
            len(self.nodes),
            self.elements,
            self.areas,
            self.qpoints,
            self.forcing,
        )

    @staticmethod
    def _default_forcing(
        x: NDArray,
    ) -> NDArray:
        """Default forcing f(x) = 100."""
        return 100 * np.ones(
            len(x),
            dtype=float,
        )

    @property
    def ncoord(self) -> int:
        """Number of mesh nodes."""
        return len(self.nodes)

    @property
    def nelem(self) -> int:
        """Number of triangular elements."""
        return len(self.elements)

    @property
    def ndof(self) -> int:
        """Number of interior degrees of freedom."""
        return len(self.interior)

    @abstractmethod
    def diffusion_coefficient(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> NDArray:
        """Evaluate the diffusion coefficient a(x, y)."""
        pass

    def solve(
        self,
        y: ArrayLike,
    ) -> FEMSolution:
        """Solve the finite-element system for parameter y."""
        y = self._validate_parameter(y)

        def coefficient(
            x: NDArray,
        ) -> NDArray:
            return self.diffusion_coefficient(x, y)

        stiffness = assemble_stiffness(
            self.ncoord,
            self.areas,
            self.local_stiffness,
            self.rows,
            self.cols,
            self.qpoints,
            coefficient,
        )

        A_II = stiffness[
            self.interior
        ][:, self.interior]

        # Prescribed boundary values
        coefficients = np.full(
            self.ncoord,
            self.boundary_data,
            dtype=float,
        )

        # Eliminate Dirichlet boundary degrees of freedom.
        rhs_I = (
            self.rhs[self.interior]
            - stiffness[
                self.interior
            ][:, self.boundary]
            @ coefficients[self.boundary]
        )

        coefficients[self.interior] = spsolve(
            A_II,
            rhs_I,
        )

        return FEMSolution(
            coefficients=coefficients,
            nodes=self.nodes,
            elements=self.elements,
            areas=self.areas,
            n=self.n,
        )


# ============================================================
# Spatially varying sine diffusion
# ============================================================

class SineDiffusion(FiniteElementParametricPDE):
    """
    FEM problem with diffusion coefficient

        a(x, y)
        = baseline
          + sum_j y_j
            sin(j*pi*x1) sin(j*pi*x2) / j^decay.
    """

    def __init__(
        self,
        dim: int = 20,
        decay: float = 1.25,
        baseline: float = 1.0,
        boundary_data: float = 0.0,
        forcing: Callable[[NDArray], NDArray] | None = None,
        level: int = 5,
    ):
        super().__init__(
            dim=dim,
            decay=decay,
            baseline=baseline,
            boundary_data=boundary_data,
            forcing=forcing,
            level=level,
        )

        self.indices = np.arange(
            1,
            self.s + 1,
            dtype=float,
        )

        self.weights = (
            self.indices ** (-self.decay)
        )

    def diffusion_coefficient(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> NDArray:
        """Evaluate the spatially varying coefficient a(x, y)."""
        x = np.atleast_2d(
            np.asarray(x, dtype=float)
        )

        y = self._validate_parameter(y)

        j = self.indices[None, :]

        modes = (
            np.sin(
                np.pi
                * x[:, 0, None]
                * j
            )
            * np.sin(
                np.pi
                * x[:, 1, None]
                * j
            )
        )

        return (
            self.baseline
            + (modes * self.weights) @ y
        )


# ============================================================
# Constant diffusion with analytical solution
# ============================================================

class ConstantDiffusion(ParametricPDE):
    """
    Problem with spatially constant affine-parametric diffusion.

    The diffusion coefficient is

        a(y) = baseline + sum_j y_j / j^decay,

    and is constant with respect to the spatial variable x.

    The PDE on D = (0, 1)^2 is

        -div(a(y) grad u(x, y)) = f(x)      in D,

    with Dirichlet boundary condition

        u(x, y) = boundary_data             on ∂D,

    where

        f(x)
        = 72 * (
            x1 * (1 - x1)
            + x2 * (1 - x2)
        ).

    The analytical solution is

        u(x, y)
        = boundary_data
        + 36 * x1(1-x1) * x2(1-x2) / a(y).

    The integral of the solution over D is

        integral_D u(x, y) dx
        = boundary_data + 1 / a(y).
    """

    def __init__(
        self,
        dim: int = 20,
        decay: float = 1.25,
        baseline: float = 1.0,
        boundary_data: float = -0.6,
    ):
        super().__init__(
            dim=dim,
            decay=decay,
            baseline=baseline,
            boundary_data=boundary_data,
        )

        indices = np.arange(
            1,
            self.s + 1,
            dtype=float,
        )

        self.weights = indices ** (-self.decay)

    def solve(
        self,
        y: ArrayLike,
    ) -> AnalyticalSolution:
        """Return the closed-form solution for parameter y."""
        y = self._validate_parameter(y)

        diffusion = (
            self.baseline
            + self.weights @ y
        )

        if diffusion <= 0.0:
            raise ValueError(
                "Diffusion coefficient must be positive."
            )

        def solution(x: NDArray) -> NDArray:
            x1 = x[:, 0]
            x2 = x[:, 1]

            return (
                self.boundary_data
                + 36.0
                * x1 * (1.0 - x1)
                * x2 * (1.0 - x2)
                / diffusion
            )

        return AnalyticalSolution(
            function=solution,
            exact_integral=(
                self.boundary_data
                + 1.0 / diffusion
            ),
        )

# ============================================================
# Parabolic spatially varying sine diffusion
# ============================================================

class ParabolicSineDiffusion(FiniteElementParametricPDE):
    """
    Time-dependent parabolic diffusion problem on D = (0, 1)^2.

    The PDE is

        ∂_t u(x, t, y)
        - div(a(x, y) grad u(x, t, y))
        = x1,

    for

        x in D,
        t in (0, T],

    with homogeneous Dirichlet boundary condition

        u(x, t, y) = 0
        on ∂D,

    and zero initial condition

        u(x, 0, y) = 0.

    The parameter vector satisfies

        y in [0, 1]^s.

    The diffusion coefficient is

        a(x, y)
        = baseline
          + sum_{j=1}^s
            y_j
            sin(j*pi*x1)
            sin(j*pi*x2)
            / j^decay.


    The returned solution is the finite-element approximation
    at final time T.

    Spatial discretization:
        P1 finite elements.

    Time discretization:
        implicit Euler,

            (M + dt A(y)) u^{k+1}
            = M u^k + dt F.

    The default forcing is

        f(x) = x1.
    """

    def __init__(
        self,
        dim: int = 20,
        decay: float = 1.25,
        baseline: float = 1.0,
        boundary_data: float = 0.0,
        level: int = 5,
        final_time: float = 1.0,
        time_steps: int = 100,
    ):
        # The forcing of the parabolic example is f(x) = x1.
        def forcing(x: NDArray) -> NDArray:
            return x[:, 0]

        super().__init__(
            dim=dim,
            decay=decay,
            baseline=baseline,
            boundary_data=boundary_data,
            forcing=forcing,
            level=level,
        )

        self.final_time = float(final_time)
        self.time_steps = int(time_steps)

        if self.final_time <= 0.0:
            raise ValueError(
                "final_time must be positive."
            )

        if self.time_steps <= 0:
            raise ValueError(
                "time_steps must be positive."
            )

        self.dt = (
            self.final_time
            / self.time_steps
        )

        self.indices = np.arange(
            1,
            self.s + 1,
            dtype=float,
        )

        self.weights = (
            self.indices ** (-self.decay)
        )

    def _validate_unit_cube_parameter(
        self,
        y: ArrayLike,
    ) -> NDArray:
        """Check that y belongs to [0, 1]^s."""
        y = self._validate_parameter(y)

        if np.any(y < 0.0) or np.any(y > 1.0):
            raise ValueError(
                "Expected y in [0, 1]^s."
            )

        return y

    def diffusion_coefficient(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> NDArray:
        """
        Evaluate

            a(x, y)
            = baseline
              + sum_j y_j
                sin(j*pi*x1)
                sin(j*pi*x2)
                / j^decay.
        """
        x = np.atleast_2d(
            np.asarray(x, dtype=float)
        )

        y = self._validate_unit_cube_parameter(y)

        j = self.indices[None, :]

        modes = (
            np.sin(
                np.pi
                * x[:, 0, None]
                * j
            )
            * np.sin(
                np.pi
                * x[:, 1, None]
                * j
            )
        )

        return (
            self.baseline
            + (modes * self.weights) @ y
        )

    def solve(
        self,
        y: ArrayLike,
    ) -> FEMSolution:
        """
        Solve the parabolic PDE up to final_time.

        The initial condition is u(x, 0) = 0 and implicit Euler
        is used for the time discretization.
        """
        y = self._validate_unit_cube_parameter(y)

        def coefficient(
            x: NDArray,
        ) -> NDArray:
            return self.diffusion_coefficient(
                x,
                y,
            )

        # Assemble A(y).
        stiffness = assemble_stiffness(
            self.ncoord,
            self.areas,
            self.local_stiffness,
            self.rows,
            self.cols,
            self.qpoints,
            coefficient,
        )

        A_II = stiffness[
            self.interior
        ][:, self.interior]

        # Mass matrix.
        #
        # assemble_load alone is not sufficient here because
        # implicit Euler requires the actual FE mass matrix.
        #
        # For P1 elements:
        #
        #   M_K = |K| / 12 * [[2,1,1],
        #                     [1,2,1],
        #                     [1,1,2]]
        #
        mass_rows = []
        mass_cols = []
        mass_data = []

        local_mass_template = np.array(
            [
                [2.0, 1.0, 1.0],
                [1.0, 2.0, 1.0],
                [1.0, 1.0, 2.0],
            ]
        )

        for element, area in zip(
            self.elements,
            self.areas,
        ):
            local_mass = (
                area
                / 12.0
                * local_mass_template
            )

            for i in range(3):
                for j in range(3):
                    mass_rows.append(
                        element[i]
                    )
                    mass_cols.append(
                        element[j]
                    )
                    mass_data.append(
                        local_mass[i, j]
                    )

        from scipy import sparse

        mass = sparse.csr_matrix(
            (
                mass_data,
                (
                    mass_rows,
                    mass_cols,
                ),
            ),
            shape=(
                self.ncoord,
                self.ncoord,
            ),
        )

        M_II = mass[
            self.interior
        ][:, self.interior]

        # For homogeneous Dirichlet conditions this is simply
        # the interior part of the load vector.
        #
        # The original parabolic problem uses boundary_data = 0.
        if self.boundary_data != 0.0:
            raise ValueError(
                "ParabolicSineDiffusion currently assumes "
                "homogeneous Dirichlet boundary data."
            )

        rhs_I = self.rhs[
            self.interior
        ]

        # Implicit Euler matrix:
        #
        #   (M + dt A) u^{k+1}
        #       = M u^k + dt F.
        system_matrix = (
            M_II
            + self.dt * A_II
        )

        # Zero initial condition.
        u = np.zeros(
            self.ndof,
            dtype=float,
        )

        for _ in range(
            self.time_steps
        ):
            rhs = (
                M_II @ u
                + self.dt * rhs_I
            )

            u = spsolve(
                system_matrix,
                rhs,
            )

        # Construct full FE coefficient vector.
        coefficients = np.zeros(
            self.ncoord,
            dtype=float,
        )

        coefficients[
            self.interior
        ] = u

        return FEMSolution(
            coefficients=coefficients,
            nodes=self.nodes,
            elements=self.elements,
            areas=self.areas,
            n=self.n,
        )

# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Analytical example: constant diffusion
    # --------------------------------------------------------

    analytical_problem = ConstantDiffusion(
        dim=4,
        decay=1.25,
        baseline=1.0,
        boundary_data=-0.6,
    )

    y = np.array([
        0.1,
        0.2,
        0.05,
        0.1,
    ])

    u = analytical_problem.solve(y)

    point = np.array([0.5, 0.5])

    print("Constant diffusion")
    print("------------------")
    print("y =", y)
    print("u(0.5, 0.5) =", u(point)[0])
    print("Integral =", u.integrate())


    # --------------------------------------------------------
    # FEM example: spatially varying sine diffusion
    # --------------------------------------------------------

    fem_problem = SineDiffusion(
        dim=4,
        decay=1.25,
        baseline=2.0,
        boundary_data=0.0,
        level=5,
    )

    u_h = fem_problem.solve(y)

    print()
    print("Sine diffusion FEM")
    print("------------------")
    print("y =", y)
    print("u_h(0.5, 0.5) =", u_h(point)[0])
    print("Integral =", u_h.integrate())

    # ----------------------------------------------------------
    # Example usage: parabolic sine diffusion
    # ----------------------------------------------------------

    parabolic_problem = ParabolicSineDiffusion(
        dim=4,
        decay=1.25,
        baseline=2.0,
        boundary_data=0.0,
        level=5,
        final_time=1.0,
        time_steps=100,
    )

    u_h_parabolic = parabolic_problem.solve(y)

    print()
    print("Parabolic sine diffusion FEM")
    print("---------------------------")
    print("y =", y)
    print("u_h(0.5, 0.5) =", u_h_parabolic(point)[0])
    print("Integral =", u_h_parabolic.integrate())

    # --------------------------------------------------------
    # Evaluate solution at several points
    # --------------------------------------------------------

    points = np.array([
        [0.25, 0.25],
        [0.50, 0.50],
        [0.75, 0.75],
    ])

    values = u_h(points)

    print()
    print("Values at multiple points:")
    print(values)


    # --------------------------------------------------------
    # Example quantity of interest
    # --------------------------------------------------------

    def qoi(problem, y):
        """
        Example QoI: integral of the PDE solution.
        """
        solution = problem.solve(y)
        return solution.integrate()

    print()
    print("QoI =", qoi(fem_problem, y))
