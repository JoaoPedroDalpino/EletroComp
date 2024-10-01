import numpy as np
import matplotlib.pyplot as plt

#Defining parameters
nx, ny = 100, 100  #grade lenght
V0 = 1.0  #potential plates
tolerance = 1e-6  #convergence parameter

V = np.zeros((nx, ny))

#position of the plates
V[ 30:70, 20] = V0  
V[30:70, 80] = -V0  

#Jacobi Method
def jacobi_update(V):
    V_new = V.copy()
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            V_new[i, j] = 0.25 * (V[i+1, j] + V[i-1, j] + V[i, j+1] + V[i, j-1])
    return V_new

# Método iterativo de Jacobi para resolver o potencial
error = 1
iterations = 0
while error > tolerance:
    V_new = jacobi_update(V)
    error = np.max(np.abs(V_new - V))
    V = V_new
    
# Função para plotar as linhas equipotenciais
def equipotential_lines(V, Lx, Ly):
    """
    Plots the equipotential lines for a given potential V, 
    for a grid spanning from -Lx/2 to Lx/2 in the x-axis and -Ly/2 to Ly/2 in the y-axis.
    """
    N = V.shape[0]
    
    # Criar a grade para plotar
    x = np.linspace(-Lx/2, Lx/2, N)
    y = np.linspace(-Ly/2, Ly/2, N)
    
    # Plotar linhas equipotenciais com preenchimento colorido
    plt.figure()
    levels = np.linspace(V.min(), V.max(), 100)  # Definir mais níveis de contorno
    plt.contourf(x, y, V, levels=levels, cmap='jet', alpha=0.8)
    plt.title(r'Equipotential Lines (Filled)')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.xlim([x.min() - 0.25, x.max() + 0.25])
    plt.ylim([y.min() - 0.25, y.max() + 0.25])
    plt.colorbar(label='Potential (V)')
    plt.savefig('projeto2/graficos/equipotencial_lines_filled.png')

#Função para o Campo Elétrico
def electric_field(V, Lx, Ly):
    """
    Plots the electric field for a given potential V, with the potential in the
    background, for a distance Lx and Ly of these axis.

    """

    N = V.shape[0]

    # Create the grid for x and y
    x = np.linspace(-Lx/2, Lx/2, N)
    y = np.linspace(-Ly/2, Ly/2, N)
    X, Y = np.meshgrid(x, y)

    # Step size for the grid
    dx = Lx / (N - 1)
    dy = Ly / (N - 1)

    # Compute the electric field (Ex, Ey) from the potential V using central differences
    Ex = np.zeros_like(V)
    Ey = np.zeros_like(V)

    # Central differences to compute the gradient of V
    Ex[:, 1:-1] = -(V[:, 2:] - V[:, :-2]) / (2 * dx)  # ∂V/∂x
    Ey[1:-1, :] = -(V[2:, :] - V[:-2, :]) / (2 * dy)  # ∂V/∂y

    # Plot the potential as a heatmap
    plt.figure()
    plt.contourf(X, Y, V, cmap='jet', alpha=0.8)
    plt.colorbar(label='Potential (V)')

    # Plot the electric field vectors as a quiver plot
    plt.quiver(X, Y, Ex, Ey, color='black')

    plt.title('Electric Field and Potential')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([x.min() - 0.25, x.max() + 0.25])
    plt.ylim([y.min() - 0.25, y.max() + 0.25])
    
    plt.savefig('projeto2/graficos/eletric_field_filled.png')

#Plotting function
equipotential_lines(V, nx, ny)
electric_field(V, nx, ny)
