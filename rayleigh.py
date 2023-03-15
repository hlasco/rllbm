import jax.numpy as jnp
from rllbm.lattice import RayleighBenardSimulation
from tqdm.rich import trange
import netCDF4

if __name__=="__main__":
    sim = RayleighBenardSimulation(64, 64, 0.71, 1e8, 0.005)
    temperature = jnp.zeros((64,64))
    temperature = temperature.at[:,0].set(
        0.5*jnp.exp(-((sim.x/sim.nx-0.5)*10)**2)
    )

    sim.initialize(temperature, tracers = [jnp.array([0.5, 0.5])])
    
    
    with netCDF4.Dataset("outputs_1.nc", "w", format='NETCDF4_CLASSIC') as f:
        f.createDimension('nx', sim.nx)
        f.createDimension('ny', sim.ny)
        f.createDimension('time', None)
        
        f.createVariable('t', 'f4', 'time')
        f.createVariable('x', 'f4', 'nx')[:] = sim.x
        f.createVariable('y', 'f4', 'ny')[:] = sim.y
        
        f.createVariable('temp', 'f4', ('time', 'nx', 'ny'))
        f.createVariable('ux', 'f4', ('time', 'nx', 'ny'))
        f.createVariable('uy', 'f4', ('time', 'nx', 'ny'))

    for i in trange(20000):
        sim.step()
        x_0 = 0.5 * jnp.sin(sim.time/7.4567*jnp.pi)
        sim.temperature_bot = 0.5*jnp.cos(sim.time/50*jnp.pi)*jnp.exp(-((sim.x/sim.nx-0.5-x_0)*10)**2)* jnp.sin(sim.x/sim.nx * jnp.pi)
    
        if i%100 == 0:
            _, u, t = sim.get_macroscopics()
            k = i//1000
            with netCDF4.Dataset("outputs_1.nc", "a", format='NETCDF4_CLASSIC') as f:
                nctime = f.variables['t']
                nctime[k] = sim.time
                nctemp  = f.variables['temp']
                nctemp[k,:,:] = t
                ncuy = f.variables['ux']
                ncuy[k,:,:] = u[:,:,0]
                ncuy = f.variables['uy']
                ncuy[k,:,:] = u[:,:,1]
            


"""
from rllbm.lattice import D2Q5, D2Q9, stream, collide_NSE_TDE
from rllbm.lattice import no_slip_bc
import jax
import jax.numpy as jnp
from tqdm.rich import trange
import numpy as np
import netCDF4

N_POINTS_Y = 64
N_POINTS_X = 64
DX = 1 / (N_POINTS_Y - 1.0)


PR = 0.71
RA = 1e8
GR = 0.005
BUOYANCY = jnp.array([GR, 0.0])

THOT = 0*0.5
TCOLD = -0*0.5
T0 = 0.5 * (THOT + TCOLD)

DT = (GR * DX)**0.5
NU = (PR / RA)**0.5*DT/(DX*DX)
K = (1.0 / (PR * RA))**0.5*DT/(DX*DX)

OMEGANS = 1. / (3 * NU + 0.5); # Relaxation parameter for fluid
OMEGAT  = 1. / (3. * K + 0.5); # Relaxation parameter for temperature


VISUALIZE = True
PLOT_EVERY_N_STEPS = int(1.0 / DT)
N_ITERATIONS = int(2*180 / DT)
SKIP_FIRST_N_ITERATIONS = 0

d2q5 = D2Q5()
d2q9 = D2Q9()

def main():
    jax.config.update("jax_enable_x64", True)

    # Define a mesh
    x = jnp.arange(N_POINTS_X)/N_POINTS_Y
    y = jnp.arange(N_POINTS_Y)/N_POINTS_Y
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    
    x_ = jnp.arange(N_POINTS_X)
    y_ = jnp.arange(N_POINTS_Y)
    X_, Y_ = jnp.meshgrid(x_, y_, indexing="ij")

    T_ini = TCOLD + (THOT-TCOLD) * (1-Y) #* (X-1)*X
    fIn = jnp.ones((N_POINTS_X, N_POINTS_Y, d2q9.size))
    fIn = fIn * d2q9.weights[jnp.newaxis, jnp.newaxis, :]
    
    tIn = T_ini[...,jnp.newaxis] * jnp.ones((N_POINTS_X, N_POINTS_Y, d2q5.size)) 
    tIn = tIn.at[1:-1,1:-1,:].set((THOT-TCOLD) * (tIn[1:-1,1:-1,:] + 0.1*np.random.rand(N_POINTS_X-2,N_POINTS_Y-2,5)-0.05))
    T_bottom = THOT + 0.5*jnp.exp(-((x-0.5)*10)**2)
    T_top = TCOLD * jnp.ones(N_POINTS_X)
    tIn = tIn.at[:, 0, :].set(T_bottom[:, jnp.newaxis])
    tIn = tIn.at[:, -1, :].set(T_top[:, jnp.newaxis])
    tIn = tIn * d2q5.weights[jnp.newaxis, jnp.newaxis, :]
    
    tracer = jnp.array([1.0, 0.5])
    dt = DT
    
    mask0 = (X_ == 0) | (X_ == N_POINTS_X-1) | (Y_ == 0) | (Y_ == N_POINTS_Y-1)
    mask1 = (X_ == 0) | (X_ == N_POINTS_X-1)

    @jax.jit
    def update(fIn, tIn, time, tracer, dt):
        
        # (2) Macroscopic Velocities
        rho = d2q9.get_moment(fIn, 0)
        uf = d2q9.get_moment(fIn, 1) / rho[..., jnp.newaxis]
        idx, idy = jnp.floor(tracer[0] / DX), jnp.floor(tracer[1] / DX)
        tracer += uf[idx.astype(int)%N_POINTS_X, idy.astype(int)%N_POINTS_Y, :] * dt
        
        time += dt
        x_0 = 0.5 * jnp.sin(time/30*np.pi)
        T_bottom = THOT + 0.5*jnp.cos(time/50*np.pi)*jnp.exp(-((x-0.5-x_0)*10)**2) #* jnp.sin(x/xmax * jnp.pi)
        
        
        fOut = fIn
        tOut = tIn
            
        fOut, tOut = collide_NSE_TDE(fOut, d2q9, OMEGANS, tOut, d2q5, OMEGAT, BUOYANCY)
        
        fOut = stream(fOut, d2q9)
        tOut = stream(tOut, d2q5)
        
        fOut = no_slip_bc(fOut, fIn, mask0, d2q9)
        tOut = no_slip_bc(tOut, tIn, mask1, d2q5)
        
        tOut = tOut.at[:,-1,4].set(
            T_top - tOut[:,-1,0] - tOut[:,-1,1] - tOut[:,-1,3] - tOut[:,-1,2]
        )
        
        tOut = tOut.at[:,0,2].set(
            T_bottom - tOut[:,0,0] - tOut[:,0,1] - tOut[:,0,4] - tOut[:,0,3]
        )

        fIn = fOut
        tIn = tOut
        
        
        return fIn, tIn, time, tracer, dt

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
        if jnp.isnan(fIn).any():
            break
        if iteration_index%PLOT_EVERY_N_STEPS == 0:
            tracers.append(tracer)
            k = iteration_index//PLOT_EVERY_N_STEPS
            rho = d2q9.get_moment(fIn, 0)
            u = d2q9.get_moment(fIn, 1) / rho[..., jnp.newaxis]
            T = d2q9.get_moment(tIn, 0)
            
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
"""