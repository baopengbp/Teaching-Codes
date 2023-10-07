#!/usr/bin/env python
# Teaching-HF-DIIS
# Author: Peng Bao <baopeng@iccas.ac.cn>

import numpy
import scipy.linalg
from pyscf import gto, scf

############### HF in PySCF to compare ##############
#mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='cc-pvdz')
mol = gto.M(atom='''
 H          2.031885   -1.178162    0.0000
 H          2.727235    0.545776   -0.0000
 C          1.856825   -0.104575   -0.0000
 C          0.568770    0.416868   -0.0000
 H          0.432847    1.496788   -0.0000
 H         -0.432847   -1.496788    0.0000
 C         -0.568770   -0.416868    0.0000
 C         -1.856825    0.104575    0.0000
 H         -2.727235   -0.545776    0.0000
 H         -2.031885    1.178162   -0.0000
''', basis='cc-pvdz')
mf = scf.RHF(mol)
mf.kernel()
print('Reference HF total energy =', mf.e_tot)
#####################################################

#################### Teaching-HF-DIIS ####################
# RHF. Only need structure information of molecule and electronic integrals 
  
#mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='cc-pvdz')
mol = gto.M(atom='''
 H          2.031885   -1.178162    0.0000
 H          2.727235    0.545776   -0.0000
 C          1.856825   -0.104575   -0.0000
 C          0.568770    0.416868   -0.0000
 H          0.432847    1.496788   -0.0000
 H         -0.432847   -1.496788    0.0000
 C         -0.568770   -0.416868    0.0000
 C         -1.856825    0.104575    0.0000
 H         -2.727235   -0.545776    0.0000
 H         -2.031885    1.178162   -0.0000
''', basis='cc-pvdz')

hcore = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')

s1e = mol.intor_symmetric('int1e_ovlp')

nao = hcore.shape[0]

eri = mol.intor('int2e').reshape(nao,nao,nao,nao)

nocc = mol.nelectron // 2

# diis ref: 1. Pulay, P. Chem. Phys. Lett. 73, 393 (1980). 2. http://vergil.chemistry.gatech.edu/notes/diis/
_err = []
_vec = []
def diis(f, errvec):
        _err.insert(0,errvec)
        _vec.insert(0,f)
        if len(_err) > 8:
            _err.pop()
            _vec.pop() 
        nd = len(_err)      
        h = numpy.ones((nd+1,nd+1))
        h[0,0] = 0.0        
        for i in range(nd):
            for j in range(i+1):
                h[i+1,j+1] = _err[i]@_err[j]
                h[j+1,i+1] = h[i+1,j+1]
        g = numpy.zeros(nd+1)
        g[0] = 1.0
        col = numpy.linalg.solve(h, g)
        fnew = numpy.zeros(f.shape)
        for i, ci in enumerate(col[1:]):
            fnew += ci * _vec[i]
        return fnew

def energy_nuc(mol):

    charges = mol.atom_charges()

    coords = mol.atom_coords()

    rr = numpy.linalg.norm(coords.reshape(-1,1,3) - coords, axis=2)

    rr[numpy.diag_indices_from(rr)] = 1e200

    e = numpy.einsum('i,ij,j->', charges, 1./rr, charges) * .5

    return e

def get_dm(fock):

    mo_energy, mo_coeff = scipy.linalg.eigh(fock, s1e)

    e_idx = numpy.argsort(mo_energy)

    mo_coeff = mo_coeff[:,e_idx]

    dm = mo_coeff[:,:nocc]@mo_coeff[:,:nocc].T

    return dm, mo_coeff

#dm = mf.get_init_guess(mol, mf.init_guess)

dm, mo_coeff = get_dm(hcore)

scf_conv = False

cycle = 0

e_tot = 0

#1 mf_diis = mf.DIIS(mf, mf.diis_file)
#4 fock_last = hcore

while not scf_conv and cycle < 50:

    dm_last = dm

    last_hf_e = e_tot

    fock = hcore + 2 * numpy.einsum('ijkl,ji->kl', eri, dm) - numpy.einsum('ijkl,jk->il', eri, dm)

    #defalt: Pulay, P. J. Comput. Chem. 3, 556 (1982).
    sdf = s1e@dm@fock
    err_vec = sdf.T - sdf   
    #1 fock = mf_diis.update(s1e, dm, fock, mf, hcore)
    #2 err_vec = mo_coeff[:, nocc:].T@fock@mo_coeff[:, :nocc]  # gradient
    #3 err_vec = (numpy.eye(nao)-s1e@dm)@fock@mo_coeff[:, :nocc]  # gradient

    #4 Pulay, P. Chem. Phys. Lett. 73, 393 (1980).
    #4 err_vec = fock - fock_last

    fock = diis(fock, err_vec.ravel())

    #4 fock_last = fock 

    e_tot = numpy.einsum('ij,ji->', hcore + fock, dm)  + energy_nuc(mol)
    print('et',e_tot)

    dm, mo_coeff = get_dm(fock)

    norm_ddm = numpy.linalg.norm(dm-dm_last)

    if abs(e_tot-last_hf_e) < 1.0E-8 and norm_ddm < 1.0E-6:

        scf_conv = True

    cycle += 1

print('Teaching-HF total energy =', e_tot, 'Cycle number=', cycle)

