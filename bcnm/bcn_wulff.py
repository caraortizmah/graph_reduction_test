from __future__ import print_function
import os, time
import subprocess
import copy
import numpy as np
import pandas as pd
import glob

from os import remove
from re import findall
from random import seed,shuffle,choice
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import euclidean
from scipy.spatial import ConvexHull
import scipy.constants as constants
from itertools import combinations,product
from math import sqrt

from ase import Atoms,Atom
from ase.atoms import symbols2numbers
from ase.neighborlist import NeighborList
from ase.utils import basestring
from ase.cluster.factory import GCD
from ase.visualize import view
from ase.io import write,read
from ase.data import chemical_symbols,covalent_radii
from ase.spacegroup import Spacegroup
from ase.build import surface as slabBuild

from pymatgen.analysis.wulff import WulffShape
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import IMolecule
from pymatgen.core.surface import SlabGenerator,generate_all_slabs
from pymatgen import Molecule

nonMetals = ['H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne',
                  'Si', 'P', 'S', 'Cl', 'Ar',
                  'Ge', 'As', 'Se', 'Br', 'Kr',
                  'Sb', 'Te', 'I', 'Xe',
                  'Po', 'At', 'Rn']
nonMetalsNumbers=symbols2numbers(nonMetals)

delta = 1e-10
_debug = False
seed(42)

def bcn_wulff_construction(symbol, surfaces, energies, size, structure,
    rounding='closest',latticeconstant=None, maxiter=100,
    center=[0.,0.,0.],stoichiometryMethod=1,np0=False,wl_method='hybridMethod',
    sampleSize=1000,totalReduced=False,coordinationLimit='half',polar=False,
    termNature='non-metal',neutralize=False,inertiaMoment=False,debug=0):
    """Function that build a Wulff-like nanoparticle.
    That can be bulk-cut, stoichiometric and reduced
    
    Args:
        symbol(Atom):Crystal structure

        surfaces[lists]: A list of list surfaces index. 

        energies[float]: A list of surface energies for the surfaces.

        size(float): The desired aproximate size.

        structure: The desired crystal structure.  Either one of the strings
        "fcc", "bcc", "sc", "hcp", "graphite"; or one of the cluster factory
        objects from the ase.cluster.XXX modules.

        rounding (optional): Specifies what should be done if no Wulff
        construction corresponds to exactly the requested size.
        Should be a string, either "above", "below" or "closest" (the
        default), meaning that the nearest cluster above or below - or the
        closest one - is created instead.

        latticeconstant (optional): The lattice constant.  If not given,
        extracted from ase.data.

        debug (optional): If non-zero, information about the iteration towards
        the right cluster size is printed.

        center((tuple)): The origin of coordinates

        stoichiometryMethod: Method to transform Np0 in Np stoichometric 0 Bruno, 1 Danilo

        np0(bool): Only gets the Np0, by means, the one that is build by plane replication

        wl_method(string): Method to calculate the plane contributuion. Two options are
        available by now, surfaceBased and distanceBased being the first one the most
        robust solution.

        sampleSize(float): Number of selected combinations

        totalReduced(bool): Removes all unbounded and singly coordinated atoms

        coordinationLimit(int): fathers minimum coordination 

        polar(bool): Reduce polarity of the Np0

        termNature(str): Terminations, could be 'metal' or 'non-metal'

        neutralize(bool): True if hydrogen or OH is added, else False

    """
    global _debug
    _debug = debug

    if debug:
        if type(size) == float:
            print('Wulff: Aiming for cluster with radius %i Angstrom (%s)' %
                  (size, rounding))
        elif type(size) == int:
            print('Wulff: Aiming for cluster with %i atoms (%s)' %
                  (size, rounding))

        if rounding not in ['above', 'below', 'closest']:
            raise ValueError('Invalid rounding: %s' % rounding)
    # Interpret structure, if it is a string.
    if isinstance(structure, basestring):
        if structure == 'fcc':
            ##STRUCTURE INHERITS FROM CLASSES IN FACTORY
            from ase.cluster.cubic import FaceCenteredCubic as structure
        elif structure == 'bcc':
            from ase.cluster.cubic import BodyCenteredCubic as structure
        elif structure == 'sc':
            from ase.cluster.cubic import SimpleCubic as structure
        elif structure == 'hcp':
            from ase.cluster.hexagonal import HexagonalClosedPacked as structure
        elif structure == 'graphite':
            from ase.cluster.hexagonal import Graphite as structure
        elif structure == 'ext':
            from bcnm.bcn_cut_cluster import CutCluster as structure
        else:
            error = 'Crystal structure %s is not supported.' % structure
            raise NotImplementedError(error)

    # Check if the number of surfaces and the number of energies are equal
    nsurf = len(surfaces)
    if len(energies) != nsurf:
        raise ValueError('The energies array should contain %d values.'
                         % (nsurf,))

    #Calculate the interplanar distance
    recCell=symbol.get_reciprocal_cell()
    dArray=interplanarDistance(recCell,surfaces)

    # Get the equivalent surfaces
    eq=equivalentSurfaces(symbol,surfaces)
    
    #Calculate the normal normalized vectors for each surface
    norms=planesNorms(eq,recCell)

    # Get the ideal wulffPercentages
    ideal_wulff_fractions=idealWulffFractions(symbol,surfaces,energies)
    #Array for the np0 properties
    np0Properties=[]

    #This is the loop to get the NP closest to the desired size

    if len(energies) == 1:
        #For systems with only one surface energy, we dont evaluate
        #too much parameters, only chemical formula and min coord 
        scale_f = np.array([0.5])
        distances = scale_f*size
        # print('distances from bcn_wulff_construction',distances)
        layers = np.array(distances/dArray)
    else:
        small = np.array(energies)/((max(energies)*2.))
        large = np.array(energies)/((min(energies)*2.))
        midpoint = (large+small)/2.
        distances = midpoint*size
        layers= distances/dArray
        # print(layers)
    if debug>0:
        print('interplanarDistances\n',dArray)
        print('layers\n',layers)
        print('distances\n',distances)
        print('surfaces\n',surfaces)
    # print('interplanarDistances\n',dArray)
    # print('layers\n',layers)
    # print('distances\n',distances)
    # print('surfaces\n',surfaces)

    # Construct the np0
    atoms_midpoint = make_atoms_dist(symbol, surfaces, layers, distances, 
                        structure, center, latticeconstant,debug)
    # Remove uncordinated atoms
    removeUnbounded(symbol,atoms_midpoint)
    # Check the minimum coordination on metallic centers
    minCoord=check_min_coord(symbol,atoms_midpoint,coordinationLimit)
    # Save midpoint
    name = atoms_midpoint.get_chemical_formula()+str(center)+"_NP0.xyz"
    write(name,atoms_midpoint,format='xyz',columns=['symbols', 'positions'])

    # Check symmetry
    # pymatgenMolecule=IMolecule(species=atoms_midpoint.get_chemical_symbols(),coords=atoms_midpoint.get_positions())
    # pga=PointGroupAnalyzer(pymatgenMolecule)
    # centrosym=pga.is_valid_op(pga.inversion_op)

    # Calculate the Wulff-like index
    if wl_method=='surfaceBased':
        # np0Properties.extend(plane_area[0])
        if np0==True:
            pass
        else:
            reduceNano(symbol,atoms_midpoint,size,sampleSize,coordinationLimit,debug)
            
        if debug>0:
            print('--------------')
            print(atoms_midpoint.get_chemical_formula())
            print('areasIndex',areasIndex)
            print('plane_area',plane_area[0])
            print('--------------')

    #* #####################################################################
    #*#####################################################################
    #* Definitive method
    elif wl_method=='hybridMethod':
        wulff_like=hybridMethod(symbol,atoms_midpoint,surfaces,layers,distances,dArray,ideal_wulff_fractions)
        if np0==True:
            pass
        elif totalReduced==True:
            pass
        elif polar==True:
            pass
        else:
            #* Stoichiometric NPS
            reduceNano(symbol,atoms_midpoint,size,sampleSize,coordinationLimit,inertiaMoment,debug)
            
def coordination(atoms,debug,size,n_neighbour):
    #time_0_coord = time.time()
    """Now find how many atoms have the first coordination shell
    """
    
    newpath = './{}'.format(str(int(size)))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(newpath)
    name=atoms.get_chemical_formula()+'_NP0.xyz'
    write(name,atoms,format='xyz',columns=['symbols', 'positions'])

    nearest_neighbour= []
    if n_neighbour is None:
        for i in range(len(atoms.get_atomic_numbers())): 
            nearest_neighbour.append(np.min([x for x in atoms.get_all_distances()[i] if x>0]))
        nearest_neighbour_av = np.average(nearest_neighbour)
        for i in nearest_neighbour:
            if i > nearest_neighbour_av*1.5:
                print("EXITING: there is something strange with the distances, check NP0 for size", int(size))
                return None

    else:
        nearest_neighbour = [n_neighbour]*len(atoms.get_atomic_numbers())


    final = False
    while final == False:
        final = True
        
        C = make_C(atoms,nearest_neighbour)
        # atomicNumbers=atoms.get_atomic_numbers()
        # atomsCoordination=zip(atomicNumbers,C)

        # # for i in atomsCoordination:
        # #     if 
        
        coord=np.empty((0,5))
        for d in set(atoms.get_atomic_numbers()):
            a=np.array([d, np.mean([C[i] for i in range(len(atoms.get_atomic_numbers())) if atoms.get_atomic_numbers()[i] == d]),
                        np.max([C[i] for i in range(len(atoms.get_atomic_numbers())) if atoms.get_atomic_numbers()[i] == d]),
                        np.min([C[i] for i in range(len(atoms.get_atomic_numbers())) if atoms.get_atomic_numbers()[i] == d]),
                        chemical_symbols[d]])
            coord = np.append(coord,[a],axis=0)
        coord = coord[coord[:,4].argsort()]
        print("coord \n",coord)
        
        if check_stoich(atoms,coord) is 'stop':
            print("Exiting because the structure is nonmetal defective")
            return None

        E=[]
        for i in atoms.get_atomic_numbers():
            if int(i) in nonMetalsNumbers:
                E.append(1)
            else:
                for n in coord:
                    if i == int(n[0]):
                        E.append((int(n[2]))/2)
        # print('E',E)
        D = []
        print('atoms pre pop\n',atoms.get_chemical_formula())
        for i,j in enumerate(C):
            if j < E[i]:
                D.append(i)
        for i,j in enumerate(D):
            atoms.pop(j-i)
            nearest_neighbour.pop(j-i)
            C = np.delete(C,j-i)
        print('atoms post pop\n',atoms.get_chemical_formula())
        check_stoich(atoms)
        
        atoms_only_metal = copy.deepcopy(atoms)

        del atoms_only_metal[[atom.index for atom in atoms if atom.symbol in nonMetals ]]
        
        # del atoms_only_metal[[atom.index for atom in atoms if atom.symbol=='O']]
        # del atoms_only_metal[[atom.index for atom in atoms if atom.symbol=='S']]
        center_of_metal = atoms_only_metal.get_center_of_mass()
        
        S = None
        
        dev_s = 100000000.0
        index=0
        if debug == 1:
            print("Atoms to be removed",atoms.excess)
        if atoms.excess is not None:
            """
            atoms.excess is an atribute vector that contains the atom excess to be 
            stoichiometric. e.g. a (IrO2)33 initial nano contains 33 Ir and 80 O atoms
            so to be stoichiometric we have to remove 18 O, this is representated
            by the vector atom.excess [0 14].
            From here until final the stuff is:
            Calculate coordination aka C=make_C
            Generate the an array of shufled list aka S=make_F
            For every element of S recover j non metal atoms.
            in this case 14 metal atoms, the first 14 in tmp_l.

            Kepping in mind that eliminating atoms change the numeration order
            is mandatory to include the changes, this is done by tot.
            in the first elimination, tot =0, so eliminate the atom k
            in the second cycle, tot=1, so eliminate the k-1 atom, that 
            is equivalent to the k atom in the initial atoms object.

            """
            print('int(coord[i,0])',int(coord[0,0]))
            # time.sleep(10)
            for i,j in enumerate(atoms.excess):
                # print(i,j,'i,j')
                if j > 0:
                    C = make_C(atoms,nearest_neighbour)
                    S = make_F(atoms,C,nearest_neighbour,debug)
                    # print('s\n',S)
                    # S =make_F_Test(atoms,coordination_testing)
                    # break
                    for h in range(len(S)):
                        atoms1 = copy.deepcopy(atoms)
                        C1 = copy.deepcopy(C)
                        E1 = copy.deepcopy(E)
                        nearest_neighbour1 = copy.deepcopy(nearest_neighbour)
                        ind=0
                        tmp_l = []
                        for m in S[h]:
                            """
                            This part could be made more pythonic
                            """
                            if ind < int(j):
                                if atoms1.get_atomic_numbers()[m] == int(coord[i,0]):
                                    tmp_l.append(m)
                                    ind = ind + 1
                        # print('tmp_l',tmp_l)
                        tot = 0
                        for k in sorted(tmp_l):
                            # print('tot',tot)
                            atoms1.pop(k-tot)
                            nearest_neighbour1.pop(k-tot)
                            
                            C1 = np.delete(C1,k-tot)
                            E1 = np.delete(E1,k-tot)
                            tot += 1
                        # time.sleep(10)
                        atoms_only_oxygen = copy.deepcopy(atoms1)
                        del atoms_only_oxygen[[atom.index for atom in atoms1 if atom.symbol not in nonMetals]]
                        center_of_oxygen = atoms_only_oxygen.get_center_of_mass()
                        dev = np.std(abs(center_of_metal-atoms_only_oxygen.get_center_of_mass()))
                        dev_p = float("{:.7f}".format(round(float(dev*100),7)))
                        """
                        THIS WRITE IS FOR TESTING PURPOSES
                        
                        if dev_p == 0.0:
                            index += 1
                            name = atoms1.get_chemical_formula()+'_NPtest'+str(index)+".xyz"
                            write(name,atoms1,format='xyz',columns=['symbols', 'positions'])
                        """
                        if debug == True:
                            # comment = ("DC = "+str(dev*100)+' Lattice="' +
                            #            ' '.join([str(x) for x in np.reshape(atoms.cell.T,
                            #                             9, order='F')]) +
                            #            '"')
                            comment = ("DC = "+str(dev*100))
                            name = atoms1.get_chemical_formula()+'_'+str(dev_p)+".xyz"
                            write(name,atoms1,format='xyz',comment=comment,columns=['symbols', 'positions'])
                        if dev < dev_s:
                            dev_s = dev
                            atoms_f = copy.deepcopy(atoms1) 
                            nearest_neighbour_f = copy.deepcopy(nearest_neighbour1)
                            C_f = copy.deepcopy(C1)
                            E_f = copy.deepcopy(E1)
                            if debug == False:
                                if round(dev_s,7) == 0.:
                                    break
            atoms = copy.deepcopy(atoms_f)
            nearest_neighbour = copy.deepcopy(nearest_neighbour_f)
            C = copy.deepcopy(C_f)
            E = copy.deepcopy(E_f)

        C = make_C(atoms,nearest_neighbour)

        for i in range(len(atoms.get_atomic_numbers())):
            if C[i] < E[i]:
                final = False
                print("final",final)
                break
          

    coord_final=np.empty((0,2))
    for d in set(atoms.get_atomic_numbers()):
        a=np.array([d, np.mean([C[i] for i in range(len(atoms.get_atomic_numbers())) if atoms.get_atomic_numbers()[i] == d])])
        coord_final = np.append(coord_final,[a],axis=0)
    
    check_stoich(atoms)
    if atoms.stoichiometry == True:
        
        if S == None:
            if debug == 1:
                print("It did't go into the coordination loop, check why")
            atoms_only_oxygen = copy.deepcopy(atoms)
            del atoms_only_oxygen[[atom.index for atom in atoms if atom.symbol not in nonMetals]]
            # del atoms_only_oxygen[[atom.index for atom in atoms if atom.symbol!='O']]
            center_of_oxygen = atoms_only_oxygen.get_center_of_mass()
            
            dev_s = np.std(abs(center_of_metal-center_of_oxygen))
            dev_p = float("{:.7f}".format(round(float(dev_s*100),7)))

        else:
            if debug == 1:
                print(len(S)," different combinations were tried resulting in", 
                      len([name for name in os.listdir('.') if os.path.isfile(name)])-1,"final NPs")
                """
                Identify the equal models with sprint coordinates
                """
                # print (os.listdir('.'))
                singulizator(glob.glob('*.xyz'),debug)


        dev_p = float("{:.7f}".format(round(float(dev_s*100),7)))
        name = atoms.get_chemical_formula()+'_NPf_'+str(dev_p)+".xyz"
        # comment = ("DC = "+ str(dev_s*100) +
        #        str(np.reshape(coord_final,(1,4))) +
        #        ' Lattice="' +
        #        'a= '+str(atoms.cell[0,0])+
        #        ' b= '+str(atoms.cell[1,1])+
        #        ' c= '+str(atoms.cell[2,2]) +
        #        '"')
        comment = ("DC = "+ str(dev_s*100)) 

        print("Final NP", atoms.get_chemical_formula(), "| DC =", dev_p, "| coord", coord_final[:,0], coord_final[:,1])
        write(name,atoms,format='xyz',comment=comment,columns=['symbols', 'positions'])
        return atoms
  
def singulizator(nanoList,debug):
    """
    Function that eliminates the nanoparticles
    that are equivalent by SPRINT coordinates
    """

    print('Enter in the singulizator')
    time_F0 = time.time()

    sprintCoordinates=[]
    results=[]
    sprintTime=time.time()
    for i in nanoList:
        # print (i)
        sprintCoordinates.append(sprint(i))
        # break
    print('end sprints',np.round((time.time()-sprintTime),2), 's')
    convStart=time.time()
    for c in combinations(range(len(sprintCoordinates)),2):
    #     # print (c[0],c[1],'c')
        if compare(sprintCoordinates[c[0]],sprintCoordinates[c[1]]) ==True:
            results.append(c)
    print('end conv',np.round((time.time()-convStart),2), 's')

    # print(results)
    
    # keeping in mind that a repeated structure can appear
    # on both columns, I just take the first
    
    if debug>0:
        for i in results:
            print('NP '+nanoList[i[0]]+' and '+nanoList[i[1]]+ ' are equal')

    
    results1=[i[0] for i in results]
    # print (results1)
    toRemove=list(set(results1))

    for i in toRemove:
        # print(i)
        # print('NP '+nanoList[results[i][0]]+' and '+nanoList[results[i][1]]+ ' are equal')
        # print('Removing '+nanoList[results[i][0]])
        remove(nanoList[i])
        # pass
    finalModels=len(nanoList)-len(toRemove)
    print('Removed NPs:',len(toRemove))
    print('Final models:',finalModels)

    time_F1 = time.time()
    print("Total time singulizator", round(time_F1-time_F0,5)," s\n")

def sprint(nano):
    """
    Function that calculates the sprint coordinates matrix for a nanoparticle.

    First calculate the coordination, then build the adjacency
    matrix. To calculate the coordination firstly generates a 
    nearest_neighbour cutoffs for NeighborList.

    The C list contains the atom and the binding atoms indices.
    From C list we build the adjMatrix. The idea is translate from
    coordination list to adjacency matrix.

    Then, calculate the sprint coordinates
    Args:
        nano(file): xyz file
    Return:
        sFormated([float]): SPRINT coordinates
    """
    atoms=read(nano,format='xyz')
    atoms.center(vacuum=20)
    adjacencyName=nano+'dat'

    # print (nano)
    nearest_neighbour=[]
    C=[]

    for i in range(len(atoms.get_atomic_numbers())):
        nearest_neighbour.append(np.min([x for x in atoms.get_all_distances()[i] if x>0]))

    half_nn = [x /2.5 for x in nearest_neighbour]
    nl = NeighborList(half_nn,self_interaction=False,bothways=True)
    nl.update(atoms)

    for i in range(len(atoms.get_atomic_numbers())):
        indices, offsets = nl.get_neighbors(i)
        C.append([i,indices])
        # print(i,indices) 

    m=len(C)
    adjMatrix=np.zeros((m,m))
    for i in C:
        for j in i[1]:
            adjMatrix[i[0],j]=1.0
    # np.savetxt('adjMatrix',adjMatrix,newline='\n',fmt='%.1f')

    # Diagonal elements defined by 1+zi/10 if i is non metal
    # and 1+zi/100 if is metal

    numbers=symbols2numbers(atoms.get_atomic_numbers())
    # print(numbers)
    for i in range(len(adjMatrix)):
        if numbers[i] <=99 :
            adjMatrix[i][i]=1+float(numbers[i])/10
        else:
            adjMatrix[i][i]=1+float(numbers[i])/100

    # np.savetxt(adjacencyName,adjMatrix,newline='\n',fmt='%.3f')

    # Calculating the largest algebraic eigenvalues and their 
    # correspondent eigenvector
    val,vec=eigsh(adjMatrix,k=1,which='LA')

    # print(val,'val')
    # print(vec)
    
    # Sorting and using positive eigenvector values (by definition)
    # to calculate the sprint coordinates

    vecAbs=[abs(i)for i in vec]
    vecAbsSort=sorted(vecAbs)
    s=[sqrt(len(adjMatrix))*val[0]*i[0] for i in vecAbsSort]
    sFormated=['{:.3f}'.format(i) for i in s]
    # print (s)
    # print(sFormated)
    # print (len(s))
    return sFormated

def compare(sprint0,sprint1):

    """
    compare the SPRINT coordinates between two nanoparticles.
    If two NP has the same sprint coordinates, both are equally
    connected.
    Args:
        sprint0(list): sprint coordinates list
        sprint1(list): sprint coordinates list
    Return:
        (Bool)
    """
    # print(sprint0,'\n',sprint1) 
    # diff=(list(set(sprint0) - set(sprint1)))
    if len(sprint0)==len(sprint1):
        diff=(list(set(sprint0) - set(sprint1)))
        if len(diff)==0:
            return True

def reduceNano(symbol,atoms,size,sampleSize,coordinationLimit,inertiaMoment,debug=0):
    """
    Function that make the nano stoichiometric
    by removing dangling atoms. It is element
    insensitive
    Args:
        symbol(Atoms): Atoms object of bulk material
        atoms(Atoms): Atoms object of selected Np0
        size(float): size of nanoparticle
        sampleSize: number of replicas
        coordinationLimit(str): half or minus2
        debug(int): for print stuff

    """
    print('Enter to reduceNano')
    time_F0 = time.time()


    # Check the stoichiometry of NP0
    if check_stoich_v2(symbol,atoms,debug) is 'stop':
        print("Exiting because the structure can not achieve stoichiometry by removing just one type of ions")
        print("-------------------------")
        return None
    if check_stoich_v2(symbol,atoms,debug) is 'stoichiometric':
        print("NP0 is stoichiometric")
        print("-------------------------")
        name=atoms.get_chemical_formula()+'stoich_f.xyz'
        write(name,atoms,format='xyz',columns=['symbols', 'positions'])
        return None

    # Save as np0_f to distinguish between them and the others 
    name=atoms.get_chemical_formula()+'_NP0_f.xyz'
    write(name,atoms,format='xyz',columns=['symbols', 'positions'])

    ##Recalculate coordination after removal
    C=coordinationv2(symbol,atoms)
    # print('C',C)


    # if debug>0:
    #     atomsBeta=copy.deepcopy(atoms)
    #     for j in C:
    #         if atomsBeta[j[0]].symbol=='Cu':
    #             if len(j[1])==1:
    #                 atomsBeta[j[0]].symbol='Mg'
    #     write('coordinationEvaluation.xyz',atomsBeta)
    #     # print(*C, sep='\n')

    
    #* 4 lists:
    #* singly: contains the indexes of singly coordinated atoms
    #* father: contains the heavy metal atoms which singly
    #* coordinated atoms are bounded
    #* coordFather: contains the coordination of father atoms
    #* fatherFull: contains the combination of father and their coordination.
    
    singly=[i for i in range(len(atoms)) if len(C[i][1])==1]
    # print('singly test')
    # for i in singly:
    #     print(atoms[i].symbol)

    father=list(set([C[i][1][0] for i in singly]))

    coordFather=[len(C[i][1]) for i in father]

    fatherFull=[[i,coordFather[n]] for n,i in enumerate(father)]

    #* Add the excess attribute to atoms object
    #* and checking if the dangling atoms belong
    #* to the excess element. If not, stop
    #* and removing this nps_f
    danglingElement=check_stoich_v2(symbol,atoms,singly,debug)
    if danglingElement=='stop it':
        remove(name)
        return None

    #* if the nano does not have dangling and not stoichiometric, discard 
    #* the model
    
    # if len(singly)==0:
    #     print('NP0 does not have singly coordinated atoms to remove','\n',
    #         'to achive the stoichiometry')
    #     return None 

    if debug>0:
        print('singly:',singly)
        print('father:',father)
        print('coordFather:',coordFather)
        print('fatherFull:',fatherFull)
    #* allowedCoordination must to be generalized
    #* the upper limit is maximum coordination -2
    #* and the inferior limit is the maximum
    #* coordination. i.e. for fluorite, the maximum coordination
    #* is 8, so using list(range(8,6,-1)) we obtain the list
    #* [8, 7, 6, 5, 4] that is fully functional.
    
    maxCord=int(np.max(coordFather))
    # if maxCord == 2:
    #     mid=int(maxCord-2)
    # else:
    # print ('maxCord',maxCord)

    # coordinationLimit Evaluation
    # Default value
    if coordinationLimit=='half':
        mid=int(maxCord/2)
    elif coordinationLimit=='minus2': 
        # User value
        mid=int(maxCord-3)
    # Control the value of coordinationLimit
    # if mid > maxCord or mid<=0:
    #     print('reduction limit must be lower than the maximum coordination,',
    #     'positive, and larger than 0')
    #     return None
    # print('mid',mid)
    # exit(1)

    allowedCoordination=list(range(maxCord,mid,-1))
    print('allowedCoordination',allowedCoordination)
    # exit(1)
    if debug>0:
        print('allowedCoordinations')
        print('excess:',atoms.excess)
        print('sampleSize:',sampleSize)
    # Discard models where can not remove singly coordinated
    if np.min(coordFather) < np.min(allowedCoordination):
        print('We can not remove dangling atoms with the available coordination limits')
        print("-------------------------")
        # exit(1)
        return None
    # To have a large amounth of conformation we generate
    # 1000 replicas for removing atoms. 
    # To make the selection random we use shuffle and 
    # choice. 
    # S=xaviSingulizator(C,singly,father,fatherFull,atoms.excess,allowedCoordination)
    S=daniloSingulizator(C,singly,father,fatherFull,atoms.excess,allowedCoordination,sampleSize)
    if S==None: 
        return None
    # Build the nanoparticles removing the s atom list. Then, calculate the DC

    atomsOnlyMetal=copy.deepcopy(atoms)
    del atomsOnlyMetal[[atom.index for atom in atomsOnlyMetal if atom.symbol in nonMetals]]
    centerOfMetal = atomsOnlyMetal.get_center_of_mass()
    # print('centerOfMetal',centerOfMetal)

    #Calculate the size as the maximum distance between cations
    # npFinalSize=np.amax(atomsOnlyMetal.get_all_distances())
    #Calculate the size as the maximum distance between atoms
    npFinalSize=np.amax(atoms.get_all_distances())

    print('stoichiometric NPs:',len(S))

    nanoList=[]
    for n,s in enumerate(S):
        NP=copy.deepcopy(atoms)
        s.sort(reverse=True)
        del NP[[s]]

        #DC calculation
        atomsOnlyNotMetal = copy.deepcopy(NP)
        del atomsOnlyNotMetal[[atom.index for atom in atomsOnlyNotMetal if atom.symbol not in nonMetals]]
        centerOfNonMetal = atomsOnlyNotMetal.get_center_of_mass()
        # print('centerOfNotMetal',centerOfNonMetal)
        dev = np.std(abs(centerOfMetal-centerOfNonMetal))
        dev_p = float("{:.7f}".format(round(float(dev*100),7)))
        name=str(NP.get_chemical_formula(mode='hill'))+'_'+str(dev_p)+'_'+str(n)+'_f.xyz'
        # print('name',name)
        nanoList.append(name)
        #Saving NP
        write(name,NP,format='xyz')
        #calculating coulomb energy
        #calculating real dipole moment
        # coulomb_energy=coulombEnergy(symbol,NP)
        # # print('coulomb_energy',coulomb_energy)
        # dipole_moment=dipole(NP)
        # size as the maximum distance between cations
        # comment='E:'+str(coulomb_energy)+',mu:'+str(dipole_moment)+'size:'+str(npFinalSize)
        #replace the ase standard comment by our comment
        # command='sed -i \'2s/.*/'+comment+'/\' '+name
        # print(command)
        # subprocess.run(command,shell=True)
        # view(NP)
        # break

    time_F1 = time.time()
    print("Total time reduceNano", round(time_F1-time_F0,5)," s\n")
    #Calling the singulizator function
    if len (nanoList) >1:
        if inertiaMoment==True:
            intertiaTensorSing(atoms,S,C,nanoList) 
        else:
            if npFinalSize<20.0:
                singulizator(nanoList,debug)
            else:
                pass
    else:
        pass
    
def check_stoich_v2(Symbol,atoms,singly=0,debug=0):
    """
    Function that evaluates the stoichiometry
    of a np.
    To do it compare calculate the excess
    of majoritary element. If not excess
    the Np is stoichiometric,
    else  add the excess atribute to atoms
    object.
    Args:
        Atoms(atoms): Nanoparticle
        Symbol(atoms): unit cell atoms structure
        singly(list): List of single coordinated atoms
        debug(int): debug to print stuff
    """

    #Get the stoichiometry of the unit cell

    #Get the symbols inside the cell
    listOfChemicalSymbols=Symbol.get_chemical_symbols()
    
    #Count and divide by the greatest common divisor
    chemicalElements=list(set(listOfChemicalSymbols))

    # put the stuff in order, always metals first

    if chemicalElements[0] in nonMetals:
        chemicalElements.reverse()

    counterCell=[]
    for e in chemicalElements:
        counterCell.append(listOfChemicalSymbols.count(e))

    gcd=np.gcd.reduce(counterCell)

    cellStoichiometry=counterCell/gcd
    # Compare the cell stoichiometry with the np stoichiometry
    
    # Get the nano stoichiometry
    listOfChemicalSymbolsNp=atoms.get_chemical_symbols()
    
    #Count and divide by the greatest common divisor
    counterNp=[]
    for e in chemicalElements:
        # print(e)
        counterNp.append(listOfChemicalSymbolsNp.count(e))
    # print(counterNp)
    gcdNp=np.gcd.reduce(counterNp)

    nanoStoichiometry=counterNp/gcdNp
    # print('nanoStoichiometry:',nanoStoichiometry)
    
    # ###
    # # The nanoparticle must respect the same ratio of ions in the crystal
    # # Just one of the ions can be excesive
    # ## Test one, just the largest in proportion is in excess

    # Get the index of the maximum value of nano stoichiometry
    # that is the element that is in excess
    excesiveIonIndex=np.argmax(nanoStoichiometry)


    ## calculate how many atoms has to be removed
    excess=np.max(counterNp)-np.min(counterNp)*(np.max(cellStoichiometry)/np.min(cellStoichiometry))

    ## verify that the number of excess are larger or equal to singly
    if debug>0:
        print('values')
        print(np.max(counterNp),np.min(counterNp),np.max(cellStoichiometry))
        print(chemicalElements[excesiveIonIndex])
        print('cellStoichiometry',cellStoichiometry)
        print('nanoStoichiometry',nanoStoichiometry)
        print(excess)

    if excess==0:
        return 'stoichiometric'
    if singly !=0:
        if len([i for i in singly if atoms[i].symbol==chemicalElements[excesiveIonIndex]])<excess:
            print('NP0 does not have enough singly coordinated excess atoms to remove','\n',
                'to achive the stoichiometry for this model')
            print("-------------------------")
            return 'stop it'

    elif excess<0 or excess%1!=0:
        return 'stop'
    else:
        atoms.excess=excess
    # if singly !=0:
    #     # print('holaaaaa')
    #     # print('singly del none',singly)
    #     # test=[i for i in singly]
    #     # print('test',test)

    #     if len([i for i in singly if atoms[i].symbol==chemicalElements[excesiveIonIndex]])<excess:
    #         print('NP0 does not have enough singly coordinated excess atoms to remove','\n',
    #             'to achive the stoichiometry for this model')
    #         return 'stop'
   

        # print('atoms excess',atoms.excess)
    
    # if excess==0:
    #     return 'stoichiometric'
    # elif excess<0 or excess%1!=0:
    #     return 'stop'

def coordinationv2(symbol,atoms):
    """
    function that calculates the
    coordination based on cutoff
    distances from the crystal,
    the distances was calculated
    by using the MIC
    Args:
        symbol(atoms): atoms object of the crystal
        atoms(atoms): atoms object for the nano
    """
    # get the neigboors for the crystal object by
    # element and keeping the maxima for element
    # as unique len neighbour

    red_nearest_neighbour=[]
    distances=symbol.get_all_distances(mic=True)
    elements=list(set(symbol.get_chemical_symbols()))
    # print(elements)
    for i in elements:
        # print(i)
        nearest_neighbour_by_element=[]
        for atom in symbol:
            if atom.symbol ==i:
                nearest_neighbour_by_element.append(np.min([x for x in distances[atom.index] if x>0]))
        # print(list(set(nearest_neighbour_by_element)))
        red_nearest_neighbour.append(np.max(nearest_neighbour_by_element))
    # print('red_nearest')
    # print(red_nearest_neighbour)

    #construct the nearest_neighbour for the nano
    nearest_neighbour=[]
    for atom in atoms:
        for n,element in enumerate(elements):
            if atom.symbol==element:
                # print('n',n)
                nearest_neighbour.append(red_nearest_neighbour[n])
    # print('nearest')
    # print(nearest_neighbour) 
    C=[]
    half_nn = [x /2.5 for x in nearest_neighbour]
    nl = NeighborList(half_nn,self_interaction=False,bothways=True)
    nl.update(atoms)
    for i in range(len(atoms.get_atomic_numbers())):
        indices, offsets = nl.get_neighbors(i)
        C.append([i,indices])
    return C
    # 
    # print(C)

def intertiaTensorSing(atoms,S,C,nanoList):
    """
    Function that uses intertia tensor to 
    compare equivalent structures
    Args: 
        atoms(Atoms): NP0 Nanoparticle
        S([]): lists of atoms to remove to achieve stoichiometry
        C([]): Coordination 
    Return:
        finalNanos([]): unique list of atoms to remove by rotational
    """
    # Get the fathers index and the eigenvalues of the inertia tensor
    start=time.time()
    # fathers=[coord[1][0] for coord in C if len(coord[1])==1]
    eigenVals=[]
    for s in S:
        # itime=time.time()
        danglingAtom=copy.deepcopy(atoms)
        toRemove=sorted([atom.index for atom in danglingAtom if atom.index not in s],reverse=True)
        del danglingAtom[toRemove]
        # write('tmp.xyz',danglingAtom,format='xyz')
        molecule=Molecule(danglingAtom.get_chemical_symbols(),danglingAtom.get_positions())
        # molecule=Molecule.from_file('tmp.xyz')
        sym=PointGroupAnalyzer(molecule)
        eigenVals.append(np.round(sym.eigvals,decimals=5))
        # print('intertia tensor calcs one',time.time()-itime,' s')    
    dataFrame=pd.DataFrame(np.array(eigenVals).reshape(len(eigenVals),3),columns=['0','1','2'])
    # duplicates=dataFrame[dataFrame.duplicated(keep=False)].sort_values(by='0')
    # print(type(duplicates))
    # duplicatesNames=[nanoList[i] for i in list(duplicates.index)] 
    # print(duplicates)
    # print(*duplicatesNames,sep='\n')
    # exit(1)
    # print(dataFrame)
    sindu=dataFrame.drop_duplicates(keep='first')

    uniqueModelsIndex=list(sindu.index)
    sample=range(len(S))
    deletedNanosIndex=[i for i in sample if i not in uniqueModelsIndex]
    end=time.time() 
    
    deleteNanos=[nanoList[i] for i in deletedNanosIndex]
    # print(*deleteNanos,sep='\n')
    for i in deleteNanos:
        remove(i) 

    print('Removed NPs:',len(deletedNanosIndex))
    # print('uniqueIndex',len(uniqueModelsIndex))
    print('Final models:',len(uniqueModelsIndex))
    
    print('Total time inertia tensor singulizator',np.round((end-start),2), 'sec')
        
