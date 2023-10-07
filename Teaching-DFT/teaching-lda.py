#!/usr/bin/env python
# Teaching-DFT
# Author: Peng Bao <baopeng@iccas.ac.cn>

import numpy
import scipy.linalg
from pyscf import gto, scf, dft

############### DFT in PySCF to compare ##############
mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='cc-pvdz')
mf = dft.RKS(mol)
mf.xc = 'lda'
mf.kernel()
print('Reference DFT total energy =', mf.e_tot)
#####################################################

#################### Teaching-DFT ####################
# RHF. Only need structure information of molecule and electronic integrals 
  
mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='cc-pvdz')

###########
grids = dft.gen_grid.Grids(mol)
grids.build()
ao = dft.numint.eval_ao(mol, grids.coords)
Cx = -3/4*(3/4/numpy.pi)**(1/3)

def get_xc(dm, ao, grids):
    rho = 2 * numpy.einsum('pu, uv, pv->p', ao, dm, ao)
    exc = 2*Cx * (rho/2)**(1/3) 
    en_xc = (rho * grids.weights)@exc
    vxc1 = 4/3*exc
    v_xc = numpy.einsum('pu,pv,p->uv', ao, ao, grids.weights*vxc1)
    return en_xc, v_xc
###########

hcore = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')

s1e = mol.intor_symmetric('int1e_ovlp')

nao = hcore.shape[0]

eri = mol.intor('int2e').reshape(nao,nao,nao,nao)

nocc = mol.nelectron // 2

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

dm, mo_coeff = get_dm(hcore)

scf_conv = False

cycle = 0

e_tot = 0

while not scf_conv and cycle < 50:

    dm_last = dm

    last_hf_e = e_tot

    en_xc, v_xc = get_xc(dm, ao, grids)

    fock = hcore + 2 * numpy.einsum('ijkl,ji->kl', eri, dm) + v_xc

    sdf = s1e@dm@fock
    err_vec = sdf.T - sdf  
    fock = diis(fock, err_vec.ravel())

    e_tot = numpy.einsum('ij,ji->', hcore + fock - v_xc, dm) + en_xc + energy_nuc(mol) 
    print('et',e_tot)

    dm, mo_coeff = get_dm(fock)

    norm_ddm = numpy.linalg.norm(dm-dm_last)

    if abs(e_tot-last_hf_e) < 1.0E-8 and norm_ddm < 1.0E-6:

        scf_conv = True

    cycle += 1

print('Teaching-DFT total energy =', e_tot, 'Cycle number=', cycle)
