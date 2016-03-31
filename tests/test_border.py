from parcels import Grid, Particle, JITParticle, AdvectionRK4, AdvectionEE
from argparse import ArgumentParser
import numpy as np
import math  # NOQA


method = {'RK4': AdvectionRK4, 'EE': AdvectionEE}


def border_grid(xdim=20, ydim=20):
    # Set NEMO grid variables
    depth = np.zeros(1, dtype=np.float32)
    time = np.zeros(1, dtype=np.float64)

    lon = np.linspace(0, 100, xdim)
    lat = np.linspace(0, 100, ydim)
    U = np.ones((xdim, ydim), dtype=np.float32)
    V = -np.ones((xdim, ydim), dtype=np.float32)
    U[:, :xdim/2] = np.nan
    V[:, :xdim/2] = np.nan

    return Grid.from_data(U, lon, lat, V, lon, lat,
                          depth, time)


def border_example(grid, npart, mode='jit', degree=1,
                   verbose=False, output=True, method=AdvectionRK4):
    # Determine particle class according to mode
    ParticleClass = JITParticle if mode == 'jit' else Particle

    # Initialise particles
    x = 3
    y = (50, 70)
    pset = grid.ParticleSet(npart, pclass=ParticleClass, start=(x, y[0]), finish=(x, y[1]))

    if verbose:
        print("Initial particle positions:\n%s" % pset)

    # Advect the particles for 24h
    time = 24 * 360000.
    dt = 36000.
    substeps = 1
    out = pset.ParticleFile(name="BorderParticle") if output else None
    print("Border: Advecting %d particles for %d timesteps"
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
    p.add_argument('-d', '--degree', type=int, default=1,
                   help='Degree of spatial interpolation')
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
        filename = 'border'
        grid = border_grid(args.grid[0], args.grid[1])
        grid.write(filename)

    # Open grid file set
    grid = Grid.from_nemo('border')

    if args.profiling:
        from cProfile import runctx
        from pstats import Stats
        runctx("border_example(grid, args.particles, mode=args.mode,\
                                   degree=args.degree, verbose=args.verbose,\
                                   output=not args.nooutput, method=method[args.method])",
               globals(), locals(), "Profile.prof")
        Stats("Profile.prof").strip_dirs().sort_stats("time").print_stats(10)
    else:
        border_example(grid, args.particles, mode=args.mode,
                       degree=args.degree, verbose=args.verbose,
                       output=not args.nooutput, method=method[args.method])
