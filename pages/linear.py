import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.integrate import solve_ivp



st.set_page_config(layout="wide")
st.markdown("""
    <style>
        /* Remove top padding */
        .block-container {
            padding-top: 2rem;
        }
        /* Optional: reduce padding between elements */
        .element-container {
            margin-top: 0rem;
            margin-bottom: 0rem;
        }
    </style>
""", unsafe_allow_html=True)
#st.title("Phase Portrait Plotter")

col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    # st.subheader("Matrix Input")

    st.latex(r"""
    \begin{pmatrix} x' \\ y' \end{pmatrix} = 
    \begin{pmatrix} a_{11}(t) & a_{12}(t) \\ a_{21}(t) & a_{22}(t) \end{pmatrix} 
    \begin{pmatrix} x \\ y \end{pmatrix} + 
    \begin{pmatrix} f_{1}(t) \\ f_{2}(t) \end{pmatrix}
    """)

    # row 1
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.caption("a₁₁(t)")
        a11_input = st.text_input("A11", value="sin(t)", label_visibility="collapsed")
    with r1c2:
        st.caption("a₁₂(t)")
        a12_input = st.text_input("A12", value="t", label_visibility="collapsed")

    # row 2
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.caption("a₂₁(t)")
        a21_input = st.text_input("a21", value="exp(t)", label_visibility="collapsed")
    with r2c2:
        st.caption("a₂₂(t)")
        a22_input = st.text_input("a22", value="cos(t)", label_visibility="collapsed")
        
    t = sp.symbols('t')
    if 't_value' not in st.session_state:
        st.session_state.t_value = 0.0

    # row 3: f(t)
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        st.caption("f₁(t)")
        f1_input = st.text_input("f₁(t)", value="0", label_visibility="collapsed")
    with r3c2:
        st.caption("f₂(t)")
        f2_input = st.text_input("f₂(t)", value="0", label_visibility="collapsed")
        
    st.markdown("### Plot an integral curve")
    plot_integral_curve = st.checkbox("Plot integral curve", value=False)

    # Initial time
    st.caption("Initial time $t_i$")
    t_i = st.number_input("Initial time", value=0.0, format="%.2f", label_visibility="collapsed")

    # Initial state vector (x(t_i), y(t_i))
    ic_col1, ic_col2 = st.columns(2)
    with ic_col1:
        st.caption(r"$x(t_i)$")
        x0 = st.number_input("x(t_i)", value=1.0, format="%.2f", label_visibility="collapsed")
    with ic_col2:
        st.caption(r"$y(t_i)$")
        y0 = st.number_input("y(t_i)", value=0.0, format="%.2f", label_visibility="collapsed")

    st.session_state.t_value = st.slider("Choose t value:", -10.0, 10.0, st.session_state.t_value, 0.1)
    
    normalize = st.checkbox("Normalize arrows", value=True)
    plot_eigenspaces = st.checkbox("Show Eigenspaces", value=True)
    
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        st.caption("x-axis scale")
        xScale = st.number_input("X-axis scale", min_value=1, max_value=50, value=5, step=1, label_visibility="collapsed")
    with r4c2:
        st.caption("y-axis scale")
        yScale = st.number_input("Y-axis scale", min_value=1, max_value=50, value=5, step=1, label_visibility="collapsed")

    # Symbolic to numeric conversion
    a11_func = sp.lambdify(t, sp.sympify(a11_input), 'numpy')
    a12_func = sp.lambdify(t, sp.sympify(a12_input), 'numpy')
    a21_func = sp.lambdify(t, sp.sympify(a21_input), 'numpy')
    a22_func = sp.lambdify(t, sp.sympify(a22_input), 'numpy')

    f1_func = sp.lambdify(t, sp.sympify(f1_input), 'numpy')
    f2_func = sp.lambdify(t, sp.sympify(f2_input), 'numpy')

    # For vector field at current t only:
    A = np.array([
        [a11_func(st.session_state.t_value), a12_func(st.session_state.t_value)],
        [a21_func(st.session_state.t_value), a22_func(st.session_state.t_value)]
    ])

    f1_val = f1_func(st.session_state.t_value)
    f2_val = f2_func(st.session_state.t_value)
    

with col_right:
    st.subheader("Phase Portrait")
    
    arrow_spacing = 0.5

    # compute number of points based on xScale and yScale
    num_points_x = max(5, int((xScale * 2) / arrow_spacing))
    num_points_y = max(5, int((yScale * 2) / arrow_spacing))

    # grid for vector field
    x_values = np.linspace(-xScale, xScale, num_points_x)
    y_values = np.linspace(-yScale, yScale, num_points_y)
    X, Y = np.meshgrid(x_values, y_values)

    U = A[0, 0] * X + A[0, 1] * Y + f1_val
    V = A[1, 0] * X + A[1, 1] * Y + f2_val

    if normalize:
        magnitude = np.sqrt(U**2 + V**2)
        magnitude[magnitude == 0] = 1e-8  # prevent dividing by zero
        U /= magnitude
        V /= magnitude

    # plot settings
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.quiver(X, Y, U, V, color='blue')
    ax.set_xlim(-xScale, xScale)
    ax.set_ylim(-yScale, yScale)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    if plot_eigenspaces:
        try:
            eigvals, eigvecs = la.eig(A)
            for vec in eigvecs.T:
                vec = vec.real  # strip imaginary part if any
                if np.allclose(vec, 0) or np.iscomplexobj(vec):
                    continue

                if abs(vec[0]) > 1e-8:
                    m = vec[1] / vec[0]
                    x = np.linspace(-xScale, xScale, 100)
                    y = m * x
                else:
                    x = np.zeros(100)
                    y = np.linspace(-yScale, yScale, 100)
                ax.plot(x, y, color='black', linewidth=1)
        except la.LinAlgError:
            st.warning("Eigenspace computation failed.")
            
    # plotting an integral curve
    if plot_integral_curve:
        def system(t_val, X_vec):
            A_t = np.array([
                [a11_func(t_val), a12_func(t_val)],
                [a21_func(t_val), a22_func(t_val)]
            ])
            f_t = np.array([f1_func(t_val), f2_func(t_val)])
            return A_t @ X_vec + f_t

        t_span = (t_i, t_i + 10)
        t_eval = np.linspace(t_span[0], t_span[1], 300)

        sol = solve_ivp(system, t_span, [x0, y0], t_eval=t_eval)

        ax.plot(sol.y[0], sol.y[1], color="red", lw=2, label="Integral curve")
        ax.scatter([x0], [y0], color="red", marker="o", label="Initial point")

    st.pyplot(fig)
    
    # Mathematical Solution Section
    if plot_integral_curve and sol is not None:
        st.subheader("Mathematical Solution")
        
        # Display the IVP formulation
        st.latex(r"""
        \text{Initial Value Problem:}\\
        \begin{cases}
        x'(t) = """ + a11_input + r""" \cdot x(t) + """ + a12_input + r""" \cdot y(t) + """ + f1_input + r"""\\
        y'(t) = """ + a21_input + r""" \cdot x(t) + """ + a22_input + r""" \cdot y(t) + """ + f2_input + r"""\\
        x(""" + str(t_i) + r""") = """ + str(x0) + r"""\\
        y(""" + str(t_i) + r""") = """ + str(y0) +
        r"""\end{cases}
        """)
        
        # Try to find analytical solution for constant coefficient case
        try:
            # Check if system has constant coefficients
            a11_sym = sp.sympify(a11_input)
            a12_sym = sp.sympify(a12_input)
            a21_sym = sp.sympify(a21_input)
            a22_sym = sp.sympify(a22_input)
            f1_sym = sp.sympify(f1_input)
            f2_sym = sp.sympify(f2_input)
            
            # Check if coefficients are constant (don't depend on t)
            is_constant = (not a11_sym.has(t) and not a12_sym.has(t) and 
                          not a21_sym.has(t) and not a22_sym.has(t) and
                          not f1_sym.has(t) and not f2_sym.has(t))
            
            if is_constant and f1_sym == 0 and f2_sym == 0:
                # Homogeneous constant coefficient system
                st.markdown("**Analytical Solution (Homogeneous Constant Coefficient System):**")
                
                # Convert to sympy matrix
                A_matrix = sp.Matrix([[a11_sym, a12_sym], [a21_sym, a22_sym]])
                
                # Find eigenvalues and eigenvectors
                eigenvals = A_matrix.eigenvals()
                eigenvects = A_matrix.eigenvects()
                
                st.write("**Matrix A:**")
                st.latex(sp.latex(A_matrix))
                
                st.write("**Eigenvalues:**")
                for eigenval, mult in eigenvals.items():
                    st.latex(f"\\lambda = {sp.latex(eigenval)} \\text{{ (multiplicity: {mult})}}")
                
                st.write("**Eigenvectors:**")
                for eigenval, mult, vects in eigenvects:
                    for i, vect in enumerate(vects):
                        st.latex(f"\\text{{For }} \\lambda = {sp.latex(eigenval)}: \\quad \\mathbf{{v}} = {sp.latex(vect)}")
                
                # General solution form
                st.write("**General Solution:**")
                x_sym, y_sym = sp.symbols('x y', cls=sp.Function)
                
                # For 2x2 systems, construct the general solution
                if len(eigenvects) == 2:  # Two distinct eigenvalues
                    eigenval1, _, vects1 = eigenvects[0]
                    eigenval2, _, vects2 = eigenvects[1]
                    v1 = vects1[0]
                    v2 = vects2[0]
                    
                    st.latex(f"""
                    \\begin{{pmatrix}} x(t) \\\\ y(t) \\end{{pmatrix}} = 
                    c_1 e^{{{sp.latex(eigenval1)} t}} {sp.latex(v1)} + 
                    c_2 e^{{{sp.latex(eigenval2)} t}} {sp.latex(v2)}
                    """)
                else:
                    st.latex(r"\text{General solution depends on the specific eigenvalue structure}")
                
            else:
                st.markdown("**Numerical Solution:**")
                st.write("This system has time-varying coefficients or non-zero forcing terms.")
                st.write("The solution is computed numerically using scipy's solve_ivp.")
                
        except Exception as e:
            st.markdown("**Numerical Solution:**")
            st.write("Analytical solution could not be computed. Using numerical integration.")
            