import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from utils.some_helpers import caret_to_exponent

st.set_page_config(layout="wide")

col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.latex(r"""
    \begin{cases}
    x' = f(t, x, y) \\
    y' = g(t, x, y)
    \end{cases}
    """)

    t, x, y = sp.symbols('t x y')

    # input nonlinear functions as strings
    f_input = st.text_input("x' =", "sin(t) - x + y")
    g_input = st.text_input("y' =", "cos(t) - y - x**2")

    # initial conditions
    plot_integral_curve = st.checkbox("Plot integral curve", value=False)
    t_i = st.number_input("Initial time $t_i$", value=0.0, format="%.2f")
    x0 = st.number_input(r"Initial $x(t_i)$", value=1.0)
    y0 = st.number_input(r"Initial $y(t_i)$", value=0.0)

    # slider for t
    if 't_value' not in st.session_state:
        st.session_state.t_value = 0.0
    st.session_state.t_value = st.slider("t", -10.0, 10.0, st.session_state.t_value, 0.1)

    # options to normalize vectors and scale axes
    normalize = st.checkbox("Normalize arrows", value=True)
    xScale = st.number_input("X-axis scale", min_value=1, max_value=50, value=5)
    yScale = st.number_input("Y-axis scale", min_value=1, max_value=50, value=5)

    # parse inputs with sympy and lambdify
    try:
        f_expr = sp.sympify(caret_to_exponent(f_input))
        g_expr = sp.sympify(caret_to_exponent(g_input))
        f_func = sp.lambdify((t, x, y), f_expr, 'numpy')
        g_func = sp.lambdify((t, x, y), g_expr, 'numpy')
    except (sp.SympifyError, Exception) as e:
        st.error(f"Error parsing expressions: {e}")
        st.stop()

with col_right:
    st.subheader("Phase Portrait")

    # arrow_spacing = 0.5
    density = 25  # increase this for more arrows
    x_vals = np.linspace(-xScale, xScale, density)
    y_vals = np.linspace(-yScale, yScale, density)
    X, Y = np.meshgrid(x_vals, y_vals)

    # vector field evaluation
    U = f_func(st.session_state.t_value, X, Y)
    V = g_func(st.session_state.t_value, X, Y)

     # Ensure U and V are numpy arrays
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    
    # Handle any NaN or infinity values
    U = np.nan_to_num(U, nan=0.0, posinf=1e6, neginf=-1e6)
    V = np.nan_to_num(V, nan=0.0, posinf=1e6, neginf=-1e6)

    if normalize:
        # Calculate magnitude
        mag = np.sqrt(U**2 + V**2)
        
        # Avoid division by zero - set very small values to a small positive number
        mag = np.where(mag < 1e-10, 1e-10, mag)
        
        # Normalize
        U_norm = U / mag
        V_norm = V / mag
        
        # Use normalized vectors
        U_plot = U_norm
        V_plot = V_norm
    else:
        U_plot = U
        V_plot = V

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.quiver(X, Y, U_plot, V_plot, color='blue')
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlim(-xScale, xScale)
    ax.set_ylim(-yScale, yScale)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    if plot_integral_curve:
        def system(t_val, X_vec):
            x_val, y_val = X_vec
            return [
                f_func(t_val, x_val, y_val),
                g_func(t_val, x_val, y_val),
            ]

        t_span = (t_i, t_i + 10)
        t_eval = np.linspace(*t_span, 300)
        sol = solve_ivp(system, t_span, [x0, y0], t_eval=t_eval)

        ax.plot(sol.y[0], sol.y[1], color='red', lw=2)
        ax.scatter([x0], [y0], color='red')

    st.pyplot(fig)