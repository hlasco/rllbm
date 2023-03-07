


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import cmasher as cmr
from tqdm import tqdm
import numpy as np
import netCDF4



N_POINTS_Y = 512
N_POINTS_X = N_POINTS_Y
DX = 1 / (N_POINTS_Y - 2.0)


PR = 0.71
RA = 1e9
GR = 0.0001
BUOYANCY = jnp.array([0, GR])

THOT = 0.5
TCOLD = -0.5
T0 = 0.5 * (THOT + TCOLD)

DT = (GR * DX)**0.5
print(DT)
NU = (PR / RA)**0.5*DT/(DX*DX)
K = (1.0 / (PR * RA))**0.5*DT/(DX*DX)

OMEGANS = 1. / (3 * NU + 0.5); # Relaxation parameter for fluid
OMEGAT  = 1. / (3. * K + 0.5); # Relaxation parameter for temperature


VISUALIZE = True
PLOT_EVERY_N_STEPS = int(0.1 / DT)
N_ITERATIONS = int(40 / DT)
SKIP_FIRST_N_ITERATIONS = 0

r"""
LBM Grid: D2Q9
    6   2   5
      \ | /
    3 - 0 - 1
      / | \
    7   4   8 
"""

N_DISCRETE_VELOCITIES = 9

D2Q9_VELOCITIES = jnp.array([
    [ 0,  1,  0, -1,  0,  1, -1, -1,  1,],
    [ 0,  0,  1,  0, -1,  1,  1, -1, -1,]
])

D2Q9_INDICES = jnp.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8,
])

OPPOSITE_D2Q9INDICES = jnp.array([
    0, 3, 4, 1, 2, 7, 8, 5, 6,
])

D2Q9_WEIGHTS = jnp.array([
    4/9,                        # Center Velocity [0,]
    1/9,  1/9,  1/9,  1/9,      # Axis-Aligned Velocities [1, 2, 3, 4]
    1/36, 1/36, 1/36, 1/36,     # 45 ° Velocities [5, 6, 7, 8]
])

RIGHT_VELOCITIES = jnp.array([1, 5, 8])
UP_VELOCITIES = jnp.array([2, 5, 6])
LEFT_VELOCITIES = jnp.array([3, 6, 7])
DOWN_VELOCITIES = jnp.array([4, 7, 8])
PURE_VERTICAL_VELOCITIES = jnp.array([0, 2, 4])
PURE_HORIZONTAL_VELOCITIES = jnp.array([0, 1, 3])

r"""
LBM Grid: D2Q5
        2
        |
    3 - 0 - 1
        |
        4 
"""

N_DISCRETE_TEMPERATURE = 5

D2Q5_TEMPERATURE = jnp.array([
    [ 0,  1,  0, -1,  0,],
    [ 0,  0,  1,  0, -1,]
])

D2Q5_INDICES = jnp.array([
    0, 1, 2, 3, 4,
])

OPPOSITE_D2Q5INDICES = jnp.array([
    0, 3, 4, 1, 2,
])

D2Q5_WEIGHTS = jnp.array([
    1/3, 1/6, 1/6, 1/6, 1/6,
])

def get_density(discrete):
    density = jnp.sum(discrete, axis=-1)
    return density

def get_macroscopic(discrete, density, lattice):
    macroscopic = jnp.einsum(
        "NMQ,dQ->NMd",
        discrete,
        lattice,
    ) / density[..., jnp.newaxis]

    return macroscopic

def get_microscopic(macroscopic, lattice):
    microscopic = jnp.einsum(
        "dQ,NMd->NMQ",
        lattice,
        macroscopic,
    )
    return microscopic

def main():
    jax.config.update("jax_enable_x64", True)

    # Define a mesh
    x = jnp.arange(N_POINTS_X)/N_POINTS_Y
    xmin = x.min()
    xmax = x.max()
    y = jnp.arange(N_POINTS_Y)/N_POINTS_Y
    
    
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    T_ini = TCOLD + (THOT-TCOLD) * (1-Y)
    fIn = jnp.ones((N_POINTS_X, N_POINTS_Y, N_DISCRETE_VELOCITIES))
    fIn = fIn * D2Q9_WEIGHTS[jnp.newaxis, jnp.newaxis, :]
    
    tIn = T_ini[...,jnp.newaxis] * jnp.ones((N_POINTS_X, N_POINTS_Y, N_DISCRETE_TEMPERATURE))+ (THOT-TCOLD) * (0.1*np.random.rand(N_POINTS_X,N_POINTS_Y,5)-0.05)
    T_bottom = THOT + (THOT-TCOLD) * (0.1*np.random.rand(N_POINTS_X)-0.05) #* jnp.exp(-((x-1)/2)**2) * jnp.sin(x/xmax * jnp.pi)/5
    #T_bottom = THOT + jnp.exp(-((x-0.5)/10)**2) * jnp.sin(x/xmax * jnp.pi) * 0
    #T_bottom = T_ini[:,0]
    tIn = tIn.at[:, 0, :].set(T_bottom[:, jnp.newaxis])
    #tIn = tIn.at[:, 0, :].set(THOT)
    #tIn = tIn.at[:, :, :].set(T_ini[...,jnp.newaxis] + (THOT-TCOLD) * (0.1*np.random.rand(N_POINTS_X,N_POINTS_Y,5)-0.05))
    #tIn = tIn.at[N_POINTS_X//2, 1, :].set(THOT + THOT/10.0)
    tIn = tIn * D2Q5_WEIGHTS[jnp.newaxis, jnp.newaxis, :]
    
    tracer = jnp.array([1.0, 0.5])
    dt = DT

    @jax.jit
    def update(fIn, tIn, time, tracer, dt):
        
        # (2) Macroscopic Velocities
        rho = get_density(fIn)
        T = get_density(tIn)
        
        uf = get_macroscopic(fIn, rho, D2Q9_VELOCITIES,)
        
        dt = 0.05 * DX / (jnp.abs(uf).max() + 0.1)
        
        NU = (PR / RA)**0.5*dt/(DX*DX)
        K = (1.0 / (PR * RA))**0.5*dt/(DX*DX)
        
        OMEGANS = 1. / (3 * NU + 0.5); # Relaxation parameter for fluid
        OMEGAT  = 1. / (3. * K + 0.5); # Relaxation parameter for temperature
        
        idx, idy = jnp.floor(tracer[0] / DX), jnp.floor(tracer[1] / DX)
        
        tracer += uf[idx.astype(int)%N_POINTS_X, idy.astype(int)%N_POINTS_Y, :] * dt
        
        time += dt
        x_0 = 0.5#*xmax * jnp.sin(time/10*np.pi)
        #T_bottom = THOT + 2*jnp.exp(-((x-0.5-x_0)/10)**2) * jnp.sin(x/xmax * jnp.pi)
        
        test = jnp.einsum(
            "dQ, d->Q",
            D2Q9_VELOCITIES,
            BUOYANCY,
        )
        
        u_norm = jnp.linalg.norm(uf, axis=-1, ord=2,)
        
        cuNS = 3*jnp.einsum(
            "dQ, NMd->NMQ",
            D2Q9_VELOCITIES,
            uf,
        )
        
        fEq =  (
            rho[..., jnp.newaxis] * D2Q9_WEIGHTS[jnp.newaxis, jnp.newaxis, :] * (
                1 + cuNS + 1/2 * cuNS**2 - 3/2 * u_norm[..., jnp.newaxis]**2
            )
        )
        force = (
            3 * rho[..., jnp.newaxis] * D2Q9_WEIGHTS[jnp.newaxis, jnp.newaxis, :] * (
                (T - T0)[..., jnp.newaxis] * test[jnp.newaxis, jnp.newaxis, :] / (THOT - TCOLD)
            )
        )
        
        cu = 3*jnp.einsum(
            "dQ, NMd->NMQ",
            D2Q5_TEMPERATURE,
            uf,
        )
        
        tEq = (
            T[..., jnp.newaxis] * D2Q5_WEIGHTS[jnp.newaxis, jnp.newaxis, :] * (
                1 + cu
            )
        )
        
        fOut = fIn - OMEGANS * (fIn-fEq) + force
        tOut = tIn - OMEGAT * (tIn-tEq)
        
        
        for i in range(N_DISCRETE_VELOCITIES):
            fOut = fOut.at[:, 0, i].set(fIn[:, 0, OPPOSITE_D2Q9INDICES[i]])
            fOut = fOut.at[:, N_POINTS_Y, i].set(fIn[:, N_POINTS_Y, OPPOSITE_D2Q9INDICES[i]])
        
        for i in range(N_DISCRETE_VELOCITIES):
            fOut = fOut.at[:, :, i].set(
                jnp.roll(
                    jnp.roll(
                        fOut[:, :, i],
                        D2Q9_VELOCITIES[0, i],
                        axis=0,
                    ),
                    D2Q9_VELOCITIES[1, i],
                    axis=1,
                )
            )
            
        for i in range(N_DISCRETE_TEMPERATURE):
            tOut = tOut.at[:,:, i].set(
                jnp.roll(
                    jnp.roll(
                        tOut[:, :, i],
                        D2Q5_TEMPERATURE[0, i],
                        axis=0,
                    ),
                    D2Q5_TEMPERATURE[1, i],
                    axis=1,
                )
            )

        tOut_ = jnp.copy(tOut)
        tOut = tOut.at[:,-1, 2].set(
            TCOLD - tOut_[:,-1,0] - tOut_[:,-1,1] - tOut_[:,-1,3] - tOut_[:,-1,4]
        )
        
        tOut = tOut.at[:,0, 4].set(
            T_bottom - tOut_[:,0,0]-tOut_[:,0,1]-tOut_[:,0,2]-tOut_[:,0,3]
        )

        fIn = fOut
        tIn = tOut
        return fIn, tIn, time, tracer, dt

    #plt.style.use("dark_background")
    #plt.figure(figsize=(10, 8), dpi=100)

    time = 0
    tracers=[]
    
    with netCDF4.Dataset("outputs_1.nc", "w", format='NETCDF4_CLASSIC') as f:
        f.createDimension('nx', N_POINTS_X)
        f.createDimension('ny', N_POINTS_Y)
        time_dim = f.createDimension('time', None)
        
        f.createVariable('t', 'f4', 'time')
        f.createVariable('x', 'f4', 'nx')[:] = x
        f.createVariable('y', 'f4', 'ny')[:] = y
        
        f.createVariable('temp', 'f4', ('time', 'nx', 'ny'))
        f.createVariable('ux', 'f4', ('time', 'nx', 'ny'))
        f.createVariable('uy', 'f4', ('time', 'nx', 'ny'))
        
        
    for iteration_index in tqdm(range(N_ITERATIONS)):
        
        if iteration_index%PLOT_EVERY_N_STEPS == 0:
            tracers.append(tracer)
            k = iteration_index//PLOT_EVERY_N_STEPS
            rho = get_density(fIn)
            T = get_density(tIn)
            
            u = get_macroscopic(fIn, rho, D2Q9_VELOCITIES,)
            
            x_0 = 0.5*xmax * jnp.sin(time/10*np.pi)
            print(jnp.abs(u).max(), dt, x_0)
            with netCDF4.Dataset("outputs_1.nc", "a", format='NETCDF4_CLASSIC') as f:
                nctime = f.variables['t']
                nctime[k] = time
                nctemp  = f.variables['temp']
                nctemp[k,:,:] = T
                ncuy = f.variables['ux']
                ncuy[k,:,:] = u[:,:,0]
                ncuy = f.variables['uy']
                ncuy[k,:,:] = u[:,:,1]
            
        fIn, tIn, time, tracer, dt = update(fIn, tIn, time, tracer, dt)
    
    #if VISUALIZE:
    #    plt.show()



if __name__ == "__main__":
    main()