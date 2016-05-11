#!/usr/bin/env python
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import matplotlib.animation as animation


def particleplotting(filename, tracerfile, recordedvar, mode):
    """Quick and simple plotting of PARCELS trajectories"""

    pfile = Dataset(filename, 'r')
    lon = pfile.variables['lon']
    lat = pfile.variables['lat']
    z = pfile.variables['z']

    if(recordedvar is not 'none'):
        record = pfile.variables[recordedvar]

    if tracerfile != 'none':
        tfile = Dataset(tracerfile,'r')
        try:
            X = tfile.variables['x']
            Y = tfile.variables['y']
        except KeyError:
            X = tfile.variables['nav_lon']
            Y = tfile.variables['nav_lat']
        try:
            P = tfile.variables['vomecrty']
        except KeyError:
            try:
                P = tfile.variables['vozocrtx']
            except KeyError:
                try:
                    P = tfile.variables['uo'][0, 0, :, :]
                    X += (X[1] - X[0]) / 2
                except KeyError:
                    P = tfile.variables['vo'][0, 0, :, :]
                    Y += (Y[1] - Y[0]) / 2
        plt.figure()
        plt.contourf(np.squeeze(X),np.squeeze(Y),np.squeeze(P))

    if mode == '3d':
        fig = plt.figure(1)
        ax = fig.gca(projection='3d')
        for p in range(len(lon)):
            ax.plot(lon[p, :], lat[p, :], z[p, :], '.-')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Depth')
    elif mode == '2d':
        plt.plot(np.transpose(lon), np.transpose(lat), '.-')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        # Quiver around landing site
#        if 1:
#            from parcels import Grid
#            basename = 'Agulhas/*'
#            grid = Grid.from_nemo(basename, vmax=1e4)
##            xi = np.where(33.68 > grid.V.lon)[0][-1] - 5
##            yi = np.where(-25.07 > grid.U.lat)[0][-1] - 5
#            xi = np.where(26.4 > grid.V.lon)[0][-1] - 20
#            yi = np.where(-33.7 > grid.U.lat)[0][-1] - 5
#            lonU = grid.U.lon[xi:xi+30]
#            latU = grid.U.lat[yi:yi+10]
#            lonV = grid.V.lon[xi:xi+30]
#            latV = grid.V.lat[yi:yi+10]
#            u = grid.U.data[0, yi:yi+10, xi:xi+30]
#            v = grid.V.data[0, yi:yi+10, xi:xi+30]
#            u[np.isnan(u)] = 0.
#            v[np.isnan(v)] = 0.
#            plt.quiver(lonU, latU, u, np.zeros(np.shape(v)))
#            plt.quiver(lonV, latV, np.zeros(np.shape(u)), v)
##            print '[%d, %d] = [%f, %f]' % (xi, yi, grid.V.lon[xi], grid.U.lat[yi])

    elif mode == 'movie2d':

        fig = plt.figure(1)
        ax = plt.axes(xlim=(np.amin(lon), np.amax(lon)), ylim=(np.amin(lat), np.amax(lat)))
        scat = ax.scatter(lon[:, 0], lat[:, 0], s=60, cmap=plt.get_cmap('autumn'))  # cmaps not working?

        def animate(i):
            scat.set_offsets(np.matrix((lon[:, i], lat[:, i])).transpose())
            if recordedvar is not 'none':
                scat.set_array(record[:, i])
            return scat,

        anim = animation.FuncAnimation(fig, animate, frames=np.arange(1, lon.shape[1]),
                                       interval=100, blit=False)

    plt.show()

if __name__ == "__main__":
    p = ArgumentParser(description="""Quick and simple plotting of PARCELS trajectories""")
    p.add_argument('mode', choices=('2d', '3d', 'movie2d'), nargs='?', default='2d',
                   help='Type of display')
    p.add_argument('-p', '--particlefile', type=str, default='MyParticle.nc',
                   help='Name of particle file')
    p.add_argument('-f', '--tracerfile', type=str, default='none',
                   help='Name of tracer file to display underneath particle trajectories')
    p.add_argument('-r', '--recordedvar', type=str, default='none',
                   help='Name of a variable recorded along trajectory')
    args = p.parse_args()

    particleplotting(args.particlefile, args.tracerfile, args.recordedvar, mode=args.mode)
