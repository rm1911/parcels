from parcels import Grid, Particle, JITParticle, AdvectionRK4, AdvectionEE
from argparse import ArgumentParser
import numpy as np


method = {'RK4': AdvectionRK4, 'EE': AdvectionEE}


def agulhas_example(grid, mode='jit', degree=1,
                   verbose=False, output=True, method=AdvectionRK4):
    # Determine particle class according to mode
    ParticleClass = JITParticle if mode == 'jit' else Particle

    # Get coastal points by subtracting the masks of u and v
    u0 = np.isnan(grid.U.data[0,:,:])
    v0 = np.isnan(grid.V.data[0,:,:])
    border = u0 - v0
    lon, lat = grid.V.lon[np.where(border)[1]], grid.U.lat[np.where(border)[0]]

    # Initialise particles
    npart = len(lon)
    pset = grid.ParticleSet(npart, pclass=ParticleClass, lon=lon, lat=lat)

    if verbose:
        print("Initial particle positions:\n%s" % pset)

    # Advect the particles for 100 days
    time = 24 * 360000.
    dt = 12000.
    substeps = 1
    out = pset.ParticleFile(name="AgulhasParticle") if output else None
    print("Agulhas: Advecting %d particles for %d timesteps"
          % (npart, int(time / dt)))

    pset.execute(method, timesteps=int(time / dt), dt=dt,
                 output_file=out, output_steps=substeps)

    if verbose:
        print("Final particle positions:\n%s" % pset)

    return pset


if __name__ == "__main__":
    p = ArgumentParser(description="""
Example of advection of particles released off the coast of South Africa and
surroundings""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='jit',
                   help='Execution mode for performing computation')
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

    basename = 'Agulhas/*'

    grid = Grid.from_nemo(basename, vmax=1e4)

    if args.profiling:
        from cProfile import runctx
        from pstats import Stats
        runctx("agulhas_example(grid, args.particles, mode=args.mode,\
                                   degree=args.degree, verbose=args.verbose,\
                                   output=not args.nooutput, method=method[args.method])",
               globals(), locals(), "Profile.prof")
        Stats("Profile.prof").strip_dirs().sort_stats("time").print_stats(10)
    else:
        agulhas_example(grid, mode=args.mode, verbose=args.verbose,
                        output=not args.nooutput, method=method[args.method])
