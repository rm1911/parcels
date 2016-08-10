from collections import OrderedDict
import numpy as np


__all__ = ['Particle', 'JITParticle', 'Variable']


class Variable(object):
    """Class encapsulating the type informationof a particular particle variable

    :param name: Name of the variable
    :param dtype: Data type of the variable
    :param size: Size of array variables
    :param alias: Allowed aliases for use in kernels. Can be a single
                  alternative name, or a list for array variables.
    """
    def __init__(self, name, dtype=np.float32, size=1, alias=None):
        self.name = name
        self.dtype = dtype
        self.size = size
        self.alias = alias
        if self.alias is not None:
            assert(isinstance(self.alias, list))
            assert(len(self.alias) == self.size)

    def __repr__(self):
        return "PVar<%s|%s%s%s>" % (self.name, self.dtype,
                                    "(%s)" % self.size if self.size > 1 else "",
                                    "::%s" % self.alias if self.alias else "")

    @property
    def itemsize(self):
        """Data type size in bytes"""
        return self.size * 8 if self.dtype == np.float64 else 4


class ParticleType(object):
    """Class encapsulating the type information for custom particles

    :param user_vars: Optional list of (name, dtype) tuples for custom variables
    """

    def __init__(self, pclass):
        if not isinstance(pclass, type):
            raise TypeError("Class object required to derive ParticleType")
        if not issubclass(pclass, Particle):
            raise TypeError("Class object does not inherit from parcels.Particle")

        self.name = pclass.__name__
        self.uses_jit = issubclass(pclass, JITParticle)
        # Ensure users variables are all of type Variable
        if isinstance(pclass.user_vars, dict):
            pclass.user_vars = [Variable(name, dtype=dtype)
                                for name, dtype in pclass.user_vars.items()]
        self.variables = pclass.base_vars + pclass.user_vars
        if self.itemsize % 8 > 0:
            # Add padding to be 64-bit aligned
            self.variables += [Variable('pad', dtype=np.float32)]
        # Build variable and alias map
        self._variable_map = {}
        self._alias_map = {}
        for v in self.variables:
            self._variable_map[v.name] = v
            if v.alias:
                self._alias_map.update([(a, (v.name, i)) for i, a in enumerate(v.alias)])

    def __repr__(self):
        return "PType<%s>::%s" % (self.name, self.variables)

    @property
    def _cache_key(self):
        return"-".join(["%s" % v for v in self.variables])

    @property
    def dtype(self):
        """Numpy.dtype object that defines the C struct"""
        return np.dtype([(v.name, (v.dtype, v.size)) for v in self.variables])

    @property
    def itemsize(self):
        """Size of the underlying particle struct in bytes"""
        return sum([v.itemsize for v in self.variables])

    @property
    def variable_map(self):
        """Alias map for particle variables of the form: alias -> (var, idx)"""
        return self._variable_map

    @property
    def alias_map(self):
        """Alias map for particle variables of the form: alias -> (var, idx)"""
        return self._alias_map


class Particle(object):
    """Class encapsualting the basic attributes of a particle

    :param lon: Initial longitude of particle
    :param lat: Initial latitude of particle
    :param grid: :Class Grid: object to track this particle on
    :param user_vars: Dictionary of any user variables that might be defined in subclasses
    """
    base_vars = []
    user_vars = []

    def __init__(self, lon, lat, grid, dt=3600., time=0., cptr=None):
        self.pos = np.array([lon, lat])
        self.time = time
        self.dt = dt

        self.xi = np.where(self.lon >= grid.U.lon)[0][-1]
        self.yi = np.where(self.lat >= grid.U.lat)[0][-1]
        self.active = 1

        for var in self.user_vars:
            setattr(self, var, 0)

    @property
    def lon(self):
        return self.pos[0]

    @lon.setter
    def lon(self, value):
        self.pos[0] = value

    @property
    def lat(self):
        return self.pos[1]

    @lat.setter
    def lat(self, value):
        self.pos[1] = value

    def __repr__(self):
        return "P(%f, %f, %f)[%d, %d]" % (self.lon, self.lat, self.time,
                                          self.xi, self.yi)

    @classmethod
    def getPType(cls):
        return ParticleType(cls)

    def delete(self):
        self.active = 0


class JITParticle(Particle):
    """Particle class for JIT-based Particle objects

    Users should extend this type for custom particles with fast
    advection computation. Additional variables need to be defined
    via the :user_vars: list of (name, dtype) tuples.

    :param user_vars: Class variable that defines additional particle variables
    """

    base_vars = [Variable('pos', dtype=np.float32, size=2, alias=['lon', 'lat']),
                 Variable('idx', np.int32, size=2, alias=['xi', 'yi']),
                 Variable('time', np.float64), Variable('dt', np.float32),
                 Variable('active', np.int32)]
    user_vars = []


    def __init__(self, *args, **kwargs):
        self._cptr = kwargs.pop('cptr', None)
        ptype = super(JITParticle, self).getPType()
        # Set up alias map
        self._alias_map = ptype.alias_map
        if self._cptr is None:
            # Allocate data for a single particle
            self._cptr = np.empty(1, dtype=ptype.dtype)[0]
        super(JITParticle, self).__init__(*args, **kwargs)

    def __getattr__(self, attr):
        if attr in ["_cptr", "_alias_map"]:
            return super(JITParticle, self).__getattr__(attr)
        elif attr in self._alias_map:
            return super(JITParticle, self).__getattr__(attr)
        else:
            return self._cptr.__getitem__(attr)

    def __setattr__(self, key, value):
        if key in ["_cptr", "_alias_map"]:
            super(JITParticle, self).__setattr__(key, value)
        elif key in self._alias_map:
            super(JITParticle, self).__setattr__(key, value)
        else:
            self._cptr.__setitem__(key, value)
