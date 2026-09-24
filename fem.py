import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve


def generate_fe_mesh(level=5):
    """Generate a uniform P1 triangulation of the unit square."""
    n = 2**level + 1
    x = np.linspace(0.0, 1.0, n)

    X, Y = np.meshgrid(x, x, indexing="xy")
    nodes = np.column_stack((X.ravel(), Y.ravel()))

    # Vectorized construction of all square-cell indices
    jj, ii = np.meshgrid(
        np.arange(n - 1),
        np.arange(n - 1),
        indexing="ij",
    )

    lower_left = (jj * n + ii).ravel()
    lower_right = lower_left + 1
    upper_left = lower_left + n
    upper_right = upper_left + 1

    # Two triangles per square
    elements = np.empty(
        (2 * lower_left.size, 3),
        dtype=np.int64,
    )

    elements[0::2] = np.column_stack((
        lower_left,
        upper_left,
        lower_right,
    ))

    elements[1::2] = np.column_stack((
        upper_left,
        upper_right,
        lower_right,
    ))

    node_ids = np.arange(n * n).reshape(n, n)
    interior = node_ids[1:-1, 1:-1].ravel()

    return nodes, elements, interior, n


def precompute_p1(nodes, elements):
    """
    Precompute quantities depending only on the mesh.

    Returns
    -------
    areas
        Triangle areas.
    local_stiffness
        Element stiffness matrices for coefficient a=1.
    rows, cols
        Global sparse-matrix indices.
    qpoints
        Degree-2 triangle quadrature points.
    """
    vertices = nodes[elements]  # (n_elements, 3, 2)

    # Triangle areas, fully vectorized
    e10 = vertices[:, 1] - vertices[:, 0]
    e20 = vertices[:, 2] - vertices[:, 0]

    twice_area = np.abs(
        e10[:, 0] * e20[:, 1]
        - e10[:, 1] * e20[:, 0]
    )

    areas = 0.5 * twice_area

    # Opposite edges for P1 basis gradients
    edges = np.stack((
        vertices[:, 2] - vertices[:, 1],
        vertices[:, 0] - vertices[:, 2],
        vertices[:, 1] - vertices[:, 0],
    ), axis=1)

    # local_stiffness[k, i, j]
    local_stiffness = (
        np.einsum("tik,tjk->tij", edges, edges)
        / (4.0 * areas[:, None, None])
    )

    # Sparse matrix row/column pattern
    rows = np.repeat(elements, 3, axis=1).ravel()
    cols = np.tile(elements, (1, 3)).ravel()

    # Three edge-midpoint quadrature points.
    # This rule is exact for polynomials of degree <= 2.
    qpoints = np.stack((
        0.5 * (vertices[:, 0] + vertices[:, 1]),
        0.5 * (vertices[:, 1] + vertices[:, 2]),
        0.5 * (vertices[:, 2] + vertices[:, 0]),
    ), axis=1)

    return areas, local_stiffness, rows, cols, qpoints


def assemble_stiffness(
    n_nodes,
    areas,
    local_stiffness,
    rows,
    cols,
    qpoints,
    coefficient,
):
    """
    Assemble

        A_ij = integral a(x) grad(phi_i) · grad(phi_j) dx

    using degree-2 triangle quadrature.

    For quadratic coefficients this integration is exact.
    """
    n_elements = len(areas)

    a_q = coefficient(
        qpoints.reshape(-1, 2)
    ).reshape(n_elements, 3)

    # Since grad(phi_i) · grad(phi_j) is constant per triangle,
    # only the element average of a is required.
    a_average = a_q.mean(axis=1)

    data = (
        a_average[:, None, None] * local_stiffness
    ).ravel()

    return sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(n_nodes, n_nodes),
    ).tocsr()


def assemble_load(
    n_nodes,
    elements,
    areas,
    qpoints,
    forcing,
):
    """
    Assemble

        b_i = integral f(x) phi_i(x) dx

    using degree-2 triangle quadrature.
    """
    n_elements = len(elements)

    f_q = forcing(
        qpoints.reshape(-1, 2)
    ).reshape(n_elements, 3)

    # Basis function values at the three edge midpoints.
    #
    # q0: midpoint vertices 0-1
    # q1: midpoint vertices 1-2
    # q2: midpoint vertices 2-0
    phi_q = np.array([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
    ])

    local_rhs = (
        areas[:, None] / 3.0
    ) * (f_q @ phi_q)

    # Fast vectorized finite-element scatter operation
    rhs = np.bincount(
        elements.ravel(),
        weights=local_rhs.ravel(),
        minlength=n_nodes,
    )

    return rhs


def evaluate_fe_uniform(points, solution, n):
    """
    Evaluate the FE solution on this particular uniform mesh.

    Unlike searching through every triangle, this finds the
    containing element directly in O(1) work per point.
    """
    points = np.atleast_2d(points).astype(float)

    values = np.full(
        len(points),
        np.nan,
        dtype=float,
    )

    valid = np.all(
        (points >= 0.0) & (points <= 1.0),
        axis=1,
    )

    if not np.any(valid):
        return values

    p = points[valid]

    h = 1.0 / (n - 1)

    # Cell containing each point
    i = np.minimum(
        (p[:, 0] / h).astype(np.int64),
        n - 2,
    )

    j = np.minimum(
        (p[:, 1] / h).astype(np.int64),
        n - 2,
    )

    # Local coordinates in the square cell
    xi = (p[:, 0] - i * h) / h
    eta = (p[:, 1] - j * h) / h

    lower_left = j * n + i
    lower_right = lower_left + 1
    upper_left = lower_left + n
    upper_right = upper_left + 1

    point_values = np.empty(len(p))

    # First triangle:
    #
    # lower_left, upper_left, lower_right
    lower_triangle = (xi + eta) <= 1.0

    k = lower_triangle

    point_values[k] = (
        (1.0 - xi[k] - eta[k])
        * solution[lower_left[k]]
        + eta[k] * solution[upper_left[k]]
        + xi[k] * solution[lower_right[k]]
    )

    # Second triangle:
    #
    # upper_left, upper_right, lower_right
    k = ~lower_triangle

    point_values[k] = (
        (1.0 - xi[k])
        * solution[upper_left[k]]
        + (xi[k] + eta[k] - 1.0)
        * solution[upper_right[k]]
        + (1.0 - eta[k])
        * solution[lower_right[k]]
    )

    values[valid] = point_values

    return values


def integrate_fe(elements, areas, solution):
    """
    Integrate a P1 FE function exactly over the mesh.
    """
    return (
        np.dot(
            areas,
            solution[elements].sum(axis=1),
        )
        / 3.0
    )

if __name__ == "__main__":
    level = 5

    # ---------------------------------------------------------
    # Mesh
    # ---------------------------------------------------------

    nodes, elements, interior, n = generate_fe_mesh(level)

    n_nodes = len(nodes)

    (
        areas,
        local_stiffness,
        rows,
        cols,
        qpoints,
    ) = precompute_p1(nodes, elements)

    # ---------------------------------------------------------
    # Problem parameters
    # ---------------------------------------------------------

    s = 4 # dimension
    baseline = 0.1
    decay = 1.25

    def forcing(x):
        return np.ones(len(x))

    parameter_samples = [
        0.1 * np.ones(s),
        0.9 * np.ones(s),
    ]

    # RHS does not depend on y
    rhs = assemble_load(
        n_nodes,
        elements,
        areas,
        qpoints,
        forcing,
    )

    point = np.array([0.5, 0.5])

    solutions = []

    # ---------------------------------------------------------
    # Solve PDE for different y
    # ---------------------------------------------------------

    for sample_number, y in enumerate(
        parameter_samples,
        start=1,
    ):

        def diffusion_at_y(x):
            """
            Parametric diffusion coefficient
                a(x, y) = baseline
                        + sum_{j=1}^s
                            y_j * sin(j*pi*x1) * sin(j*pi*x2)
                            / j**decay
            """
            x = np.asarray(x)
            j = np.arange(1, len(y) + 1)

            modes = (
                np.sin(np.pi * x[:, 0, None] * j)
                * np.sin(np.pi * x[:, 1, None] * j)
            )

            return (
                baseline
                + np.sum(
                    modes * y / j**decay,
                    axis=1,
                )
            )

        stiffness = assemble_stiffness(
            n_nodes,
            areas,
            local_stiffness,
            rows,
            cols,
            qpoints,
            diffusion_at_y,
        )

        A_int = stiffness[interior][:, interior]
        b_int = rhs[interior]

        solution = np.zeros(n_nodes)

        solution[interior] = spsolve(
            A_int,
            b_int,
        )

        solutions.append(solution)

        value_at_point = evaluate_fe_uniform(
            point,
            solution,
            n,
        )[0]

        solution_integral = integrate_fe(
            elements,
            areas,
            solution,
        )

        print(f"\nSample {sample_number}")
        print(f"y = {y}")

        print(
            f"u({point}) = "
            f"{value_at_point:.8f}"
        )

        print(
            f"Integral of u = "
            f"{solution_integral:.8f}"
        )

    # ---------------------------------------------------------
    # Plot solutions
    # ---------------------------------------------------------

    for sample_number, (y, solution) in enumerate(
        zip(parameter_samples, solutions),
        start=1,
    ):

        fig = plt.figure(
            figsize=plt.figaspect(1.0)
        )

        ax = fig.add_subplot(
            projection="3d"
        )

        ax.plot_trisurf(
            nodes[:, 0],
            nodes[:, 1],
            solution,
            triangles=elements,
            cmap="viridis",
        )

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("u")

        ax.set_title(
            f"Sample {sample_number}: y = {y}"
        )

        plt.show()