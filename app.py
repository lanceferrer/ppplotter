import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la


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


    # evaluate matrix A(t)
    A = np.array([
        [sp.lambdify(t, sp.sympify(a11_input), 'numpy')(st.session_state.t_value),
         sp.lambdify(t, sp.sympify(a12_input), 'numpy')(st.session_state.t_value)],
        [sp.lambdify(t, sp.sympify(a21_input), 'numpy')(st.session_state.t_value),
         sp.lambdify(t, sp.sympify(a22_input), 'numpy')(st.session_state.t_value)]
    ])
    
    f1_func = sp.lambdify(t, sp.sympify(f1_input), 'numpy')
    f2_func = sp.lambdify(t, sp.sympify(f2_input), 'numpy')

    f1_val = f1_func(st.session_state.t_value)
    f2_val = f2_func(st.session_state.t_value)

with col_right:
    st.subheader("Phase Plane")
    
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

    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.quiver(X, Y, U, V, color='blue')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(-xScale, xScale)
    ax.set_ylim(-yScale, yScale)
    
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

    st.pyplot(fig)