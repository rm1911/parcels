from parcels import Grid, Particle, JITParticle, AdvectionRK4, AdvectionEE
from argparse import ArgumentParser
import numpy as np
import math  # NOQA


method = {'RK4': AdvectionRK4, 'EE': AdvectionEE}


def vortex_grid(xdim=20, ydim=20):
    depth = np.zeros(1, dtype=np.float32)
    time = np.zeros(1, dtype=np.float64)

#    a = 1.1
#    b = 5.
    a = 1.1
    b = 5.
    lon = np.linspace(-2, 2, xdim)
    lat = np.linspace(-2, 2, ydim)
    U = np.nan * np.empty((xdim, ydim), dtype=np.float32)
    V = np.nan * np.empty((xdim, ydim), dtype=np.float32)
    for xi in range(xdim):
        for yi in range(ydim):
            if -1 <= lon[xi] < 1 and -1 <= lat[yi] < 1:
#            if (-1 > lon[xi] or lon[xi] >= 1) or (-1 > lat[yi] or lat[yi] >= 1):
                U[xi, yi] = a * lon[xi] - b * lat[yi] - lon[xi] * (lon[xi] ** 2 + lat[yi] ** 2)
                V[xi, yi] = b * lon[xi] + a * lat[yi] - lat[yi] * (lon[xi] ** 2 + lat[yi] ** 2)

    return Grid.from_data(U, lon, lat, V, lon, lat,
                          depth, time)


def vortex_example(grid, npart=3, mode='jit',
                   verbose=False, output=True, method=AdvectionRK4):
    # Determine particle class according to mode
    ParticleClass = JITParticle if mode == 'jit' else Particle

    # Initialise particles
    x = (0.1, 0.95)
    y = (0.1, 0.95)
#    x = (0., 0.)
#    y = (1.2, 1.9)
    pset = grid.ParticleSet(npart, pclass=ParticleClass, start=(x[0], y[0]), finish=(x[1], y[1]))

    if verbose:
        print("Initial particle positions:\n%s" % pset)

    # Advect the particles for 240h
    time = 24 * 36000.
    dt = 3600.
#    dt = 360.
    substeps = 1
    out = pset.ParticleFile(name="VortexParticle") if output else None
    print("Vortex: Advecting %d particles for %d timesteps"
          % (npart, int(time / dt)))

    pset.execute(method, timesteps=int(time / dt), dt=dt,
                 output_file=out, output_steps=substeps)

    if verbose:
        print("Final particle positions:\n%s" % pset)

    return pset


if __name__ == "__main__":
    p = ArgumentParser(description="""
Example of particle advection around an idealised peninsula""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='jit',
                   help='Execution mode for performing RK4 computation')
    p.add_argument('-p', '--particles', type=int, default=20,
                   help='Number of particles to advect')
    p.add_argument('-v', '--verbose', action='store_true', default=False,
                   help='Print particle information before and after execution')
    p.add_argument('-o', '--nooutput', action='store_true', default=False,
                   help='Suppress trajectory output')
    p.add_argument('--profiling', action='store_true', default=False,
                   help='Print profiling information after run')
    p.add_argument('-g', '--grid', type=int, nargs=2, default=None,
                   help='Generate grid file with given dimensions')
    p.add_argument('-m', '--method', choices=('RK4', 'EE'), default='RK4',
                   help='Numerical method used for advection')
    args = p.parse_args()

    if args.grid is not None:
        filename = 'vortex'
        grid = vortex_grid(args.grid[0], args.grid[1])
        grid.write(filename)

    # Open grid file set
    grid = Grid.from_nemo('vortex')

    if args.profiling:
        from cProfile import runctx
        from pstats import Stats
        runctx("vortex_example(grid, args.particles, mode=args.mode,\
                                   degree=args.degree, verbose=args.verbose,\
                                   output=not args.nooutput, method=method[args.method])",
               globals(), locals(), "Profile.prof")
        Stats("Profile.prof").strip_dirs().sort_stats("time").print_stats(10)
    else:
        vortex_example(grid, args.particles, mode=args.mode,
                       verbose=args.verbose,
                       output=not args.nooutput, method=method[args.method])
