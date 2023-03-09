

from rllbm.lattice import D2Q5, D2Q9, stream, collide
import jax
import jax.numpy as jnp
from tqdm.rich import trange
import numpy as np
import netCDF4

N_POINTS_Y = 256
N_POINTS_X = 256
DX = 1 / (N_POINTS_Y - 1.0)


PR = 0.71
RA = 1e9
GR = 0.0001
BUOYANCY = jnp.array([0, GR])

THOT = 0.5
TCOLD = -0.5
T0 = 0.5 * (THOT + TCOLD)

DT = (GR * DX)**0.5
NU = (PR / RA)**0.5*DT/(DX*DX)
K = (1.0 / (PR * RA))**0.5*DT/(DX*DX)

OMEGANS = 1. / (3 * NU + 0.5); # Relaxation parameter for fluid
OMEGAT  = 1. / (3. * K + 0.5); # Relaxation parameter for temperature


VISUALIZE = True
PLOT_EVERY_N_STEPS = int(0.1 / DT)
N_ITERATIONS = int(40 / DT)
SKIP_FIRST_N_ITERATIONS = 0

d2q5 = D2Q5()
d2q9 = D2Q9()

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

    T_ini = TCOLD + (THOT-TCOLD) * (1-Y) #* (X-1)*X
    fIn = jnp.ones((N_POINTS_X, N_POINTS_Y, d2q9.size))
    fIn = fIn * d2q9.weights[jnp.newaxis, jnp.newaxis, :]
    
    tIn = T_ini[...,jnp.newaxis] * jnp.ones((N_POINTS_X, N_POINTS_Y, d2q5.size)) 
    tIn = tIn.at[1:-1,1:-1,:].set((THOT-TCOLD) * (tIn[1:-1,1:-1,:] + 0.1*np.random.rand(N_POINTS_X-2,N_POINTS_Y-2,5)-0.05))
    #T_bottom = THOT + (THOT-TCOLD) * (0.1*np.random.rand(N_POINTS_X)-0.05) #* jnp.exp(-((x-1)/2)**2) * jnp.sin(x/xmax * jnp.pi)/5
    #T_bottom = THOT + jnp.exp(-((x-0.5)/10)**2) * jnp.sin(x/xmax * jnp.pi) * 0
    T_bottom = THOT * jnp.ones(N_POINTS_X)
    T_top = TCOLD * jnp.ones(N_POINTS_X)
    tIn = tIn.at[:, 0, :].set(T_bottom[:, jnp.newaxis])
    tIn = tIn.at[:, -1, :].set(T_top[:, jnp.newaxis])
    #tIn = tIn.at[:, 0, :].set(THOT)
    #tIn = tIn.at[:, :, :].set(T_ini[...,jnp.newaxis] + (THOT-TCOLD) * (0.1*np.random.rand(N_POINTS_X,N_POINTS_Y,5)-0.05))
    #tIn = tIn.at[N_POINTS_X//2, 1, :].set(THOT + THOT/10.0)
    tIn = tIn * d2q5.weights[jnp.newaxis, jnp.newaxis, :]
    
    tracer = jnp.array([1.0, 0.5])
    dt = DT

    @jax.jit
    def update(fIn, tIn, time, tracer, dt):
        
        # (2) Macroscopic Velocities
        rho = d2q9.get_moment(fIn, 0)
        uf = d2q9.get_moment(fIn, 1) / rho[..., jnp.newaxis]
        idx, idy = jnp.floor(tracer[0] / DX), jnp.floor(tracer[1] / DX)
        tracer += uf[idx.astype(int)%N_POINTS_X, idy.astype(int)%N_POINTS_Y, :] * dt
        
        time += dt
        x_0 = 0.5#*xmax * jnp.sin(time/10*np.pi)
        #T_bottom = THOT + 2*jnp.exp(-((x-0.5-x_0)/10)**2) * jnp.sin(x/xmax * jnp.pi)
        fOut = fIn
        tOut = tIn
            
        fOut, tOut = collide(fOut, d2q9, OMEGANS, tOut, d2q5, OMEGAT, BUOYANCY)
        
        fOut = stream(fOut, d2q9)
        tOut = stream(tOut, d2q5)
        
        for i in range(d2q9.size):
            fOut = fOut.at[:, 0, i].set(fIn[:, 0, d2q9.opposite_indices[i]])
            fOut = fOut.at[:, N_POINTS_Y, i].set(fIn[:, N_POINTS_Y,  d2q9.opposite_indices[i]])
            fOut = fOut.at[N_POINTS_X, :, i].set(fIn[N_POINTS_X, :,  d2q9.opposite_indices[i]])
            fOut = fOut.at[0, :, i].set(fIn[0, :,  d2q9.opposite_indices[i]])
            
        for i in range(d2q5.size):
            tOut = tOut.at[N_POINTS_X, :, i].set(tIn[N_POINTS_X, :,  d2q5.opposite_indices[i]])
            tOut = tOut.at[0, :, i].set(tIn[0, :,  d2q5.opposite_indices[i]])
        
        tOut = tOut.at[:,-1,4].set(
            T_top - tOut[:,-1,0] - tOut[:,-1,1] - tOut[:,-1,3] - tOut[:,-1,2]
        )
        
        tOut = tOut.at[:,0,2].set(
            T_bottom - tOut[:,0,0] - tOut[:,0,1] - tOut[:,0,4] - tOut[:,0,3]
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
        f.createDimension('time', None)
        
        f.createVariable('t', 'f4', 'time')
        f.createVariable('x', 'f4', 'nx')[:] = x
        f.createVariable('y', 'f4', 'ny')[:] = y
        
        f.createVariable('temp', 'f4', ('time', 'nx', 'ny'))
        f.createVariable('ux', 'f4', ('time', 'nx', 'ny'))
        f.createVariable('uy', 'f4', ('time', 'nx', 'ny'))
        
        
    for iteration_index in trange(N_ITERATIONS):
        
        if iteration_index%PLOT_EVERY_N_STEPS == 0:
            tracers.append(tracer)
            k = iteration_index//PLOT_EVERY_N_STEPS
            rho = d2q9.get_moment(fIn, 0)
            T = d2q9.get_moment(tIn, 0)
            
            u = get_macroscopic(fIn, rho, d2q9.coords,)
            
            #x_0 = 0.5*xmax * jnp.sin(time/10*np.pi)
            #(jnp.abs(u).max(), dt, x_0)
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



if __name__ == "__main__":
    main()