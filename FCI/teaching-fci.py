#!/usr/bin/env python
# Teaching-FCI
# Author: Peng Bao <baopeng@iccas.ac.cn>

import numpy
from pyscf import gto, scf, fci
import itertools

############### FCI in PySCF to compare ##############
mol = gto.M(atom='H 0 0 0; Be 0 0 1.1', basis='sto-3g', spin=1)
mf = scf.ROHF(mol)
mf.kernel()

cisolver = fci.FCI(mol, mf.mo_coeff)
print('E(FCI) = %.12f' % (cisolver.kernel()[0]))
######################################################

#################### Teaching-FCI ####################
def select(psi_i, psi_j):
  mask = numpy.isin(psi_i, psi_j, invert=True)
  x0a = numpy.argwhere(mask == True)
  aa0 = numpy.array(psi_i)
  nina = aa0[mask]
  mask = numpy.isin(psi_j, psi_i, invert=True)
  x0b = numpy.argwhere(mask == True)
  aa0 = numpy.array(psi_j)
  ninb = aa0[mask]
  sign = (-1)**((x0a + x0b) % 2)
  return nina, ninb, sign

mo = numpy.hstack((mf.mo_coeff,mf.mo_coeff))
nao = mo.shape[0]
noccb = (mol.nelectron - mol.spin) // 2
nocca = noccb + mol.spin

hcore = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
hcorem = numpy.einsum('ui,vj,uv->ij', mo, mo, hcore)
eri2e = mol.intor('int2e')
eri2em = numpy.einsum('ui,vj,xk,yl,uvxy->ijkl', mo, mo, mo, mo, eri2e)

deta = list(itertools.combinations(range(nao), nocca))
detb = list(itertools.combinations(range(nao), noccb))
ndeta = len(deta)
ndetb = len(detb)
ndet = ndeta * ndetb
psi = []
for i in range(ndeta): 
  for j in range(ndetb):    
    psi.append(list(deta[i]) + [k + nao for k in detb[j]])

H = numpy.zeros((ndet,ndet))
for i in range(ndet): 
  for j in range(i+1):
    orb_diffa = set(psi[i][:nocca]).difference(set(psi[j][:nocca]))
    ndiffa = len(orb_diffa)
    orb_diffb = set(psi[i][nocca:]).difference(set(psi[j][nocca:]))
    ndiffb = len(orb_diffb)
    ndiff = ndiffa + ndiffb
    if ndiff == 2:
      if ndiffa == 1:
        nina, ninb, signa = select(psi[i][:nocca], psi[j][:nocca])
        nina2, ninb2, signb = select(psi[i][nocca:], psi[j][nocca:])
        H[i,j] = eri2em[nina[0], ninb[0], nina2[0], ninb2[0]]
        H[i,j] *= signa * signb
      else:
        if ndiffa == 2: 
          nina, ninb, sign = select(psi[i][:nocca], psi[j][:nocca])
        if ndiffb == 2:   
          nina, ninb, sign = select(psi[i][nocca:], psi[j][nocca:])
        H[i,j] = eri2em[nina[0], ninb[0], nina[1], ninb[1]] - eri2em[nina[0], ninb[1], ninb[0], nina[1]]
        H[i,j] *= sign[0]*sign[1]
    else:
      ai = psi[i][:nocca]
      bi = psi[i][nocca:]
      if ndiff == 1:
        if ndiffa == 1:
          nina, ninb, sign = select(psi[i][:nocca], psi[j][:nocca])  
          H[i,j] = - numpy.einsum('nn->', eri2em[nina[0]][ai][:, ninb[0]][..., ai])
        else:
          nina, ninb, sign = select(psi[i][nocca:], psi[j][nocca:])  
          H[i,j] = - numpy.einsum('nn->', eri2em[nina[0]][bi][:, ninb[0]][..., bi])
        H[i,j] += hcorem[nina[0], ninb[0]] + numpy.einsum('nn->', eri2em[nina[0], ninb[0]][psi[i]][:,psi[i]]) 
        H[i,j] *= sign
      elif ndiff == 0:
        H[i,j] = numpy.einsum('ii->', hcorem[psi[i]][:, psi[i]]) \
               + 0.5*(numpy.einsum('iijj->', eri2em[psi[i]][:, psi[i]][:, :, psi[i]][..., psi[i]]) \
               - numpy.einsum('ijij->', eri2em[ai][:, ai][:, :, ai][..., ai]) \
               - numpy.einsum('ijij->', eri2em[bi][:, bi][:, :, bi][..., bi]))
    H[j,i] = H[i,j]

ee, ev = numpy.linalg.eigh(H)
ee += mol.energy_nuc()
print('E(FCI,20) = ', ee[:20])

