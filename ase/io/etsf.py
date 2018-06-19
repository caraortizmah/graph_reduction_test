import numpy as np

from ase.atoms import Atoms
from ase.units import Bohr


def read_etsf(filename):
    yield ETSFReader(filename).read_atoms()
    

def write_etsf(filename, atoms):
    ETSFWriter(filename).write_atoms(atoms)


class ETSFReader:
    def __init__(self, filename):
        from Scientific.IO.NetCDF import NetCDFFile
        self.nc = NetCDFFile(filename, 'r')

    def read_atoms(self):
        var = self.nc.variables
        cell = var['primitive_vectors']
        assert cell.units == 'atomic units'
        species = var['atom_species'][:]
        spos = var['reduced_atom_positions'][:]
        numbers = var['atomic_numbers'][:]
        return Atoms(numbers=numbers[species - 1],
                     scaled_positions=spos,
                     cell=cell[:] * Bohr,
                     pbc=True)
    
    
class ETSFWriter:
    def __init__(self, filename):
        from Scientific.IO.NetCDF import NetCDFFile
        self.nc = NetCDFFile(filename, 'w')

        self.nc.file_format = 'ETSF Nanoquanta'
        self.nc.file_format_version = np.array([3.3], dtype=np.float32)
        self.nc.Conventions = 'http://www.etsf.eu/fileformats/'
        self.nc.history = 'File generated by ASE'

    def write_atoms(self, atoms):
        specie_a = np.empty(len(atoms), np.int32)
        nspecies = 0
        species = {}
        numbers = []
        for a, Z in enumerate(atoms.get_atomic_numbers()):
            if Z not in species:
                species[Z] = nspecies
                nspecies += 1
                numbers.append(Z)
            specie_a[a] = species[Z]
            
        dimensions = [
            ('character_string_length', 80),
            ('number_of_atoms', len(atoms)),
            ('number_of_atom_species', nspecies),
            ('number_of_cartesian_directions', 3),
            ('number_of_reduced_dimensions', 3),
            ('number_of_vectors', 3)]

        for name, size in dimensions:
            self.nc.createDimension(name, size)

        var = self.add_variable
        
        var('primitive_vectors',
            ('number_of_vectors', 'number_of_cartesian_directions'),
            atoms.cell / Bohr, units='atomic units')
        var('atom_species', ('number_of_atoms',), specie_a + 1)
        var('reduced_atom_positions',
            ('number_of_atoms', 'number_of_reduced_dimensions'),
            atoms.get_scaled_positions())
        var('atomic_numbers', ('number_of_atom_species',),
            np.array(numbers, dtype=float))

    def close(self):
        self.nc.close()
    
    def add_variable(self, name, dims, data=None, **kwargs):
        if data is None:
            char = 'd'
        else:
            if isinstance(data, np.ndarray):
                char = data.dtype.char
            elif isinstance(data, float):
                char = 'd'
            elif isinstance(data, int):
                char = 'i'
            else:
                char = 'c'

        var = self.nc.createVariable(name, char, dims)
        for attr, value in kwargs.items():
            setattr(var, attr, value)
        if data is not None:
            if len(dims) == 0:
                var.assignValue(data)
            else:
                if char == 'c':
                    if len(dims) == 1:
                        var[:len(data)] = data
                    else:
                        for i, x in enumerate(data):
                            var[i, :len(x)] = x
                else:
                    var[:] = data
        return var
