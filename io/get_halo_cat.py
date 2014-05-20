# Last-modified: 21 Apr 2014 12:54:23

import os.path
from readsubhalo import SnapDir
import matplotlib.pyplot as plt
from zypy.zyutil import getFrames, figure_handler
import numpy as np
import itertools
import h5py
from toy import project_ellipsoid

mockdir = "/Users/ying/Data/mb2/align/"


""" Read the MB2 halo catalog.

 dark matter is 1.

"""
def test_shape(snapid, ROOT="/physics/yfeng1/mb2/", igrp=5, nptmax=1e4, set_plot=False,
        figext=None) :
    """ test shape """
    snap = SnapDir(snapid, ROOT)
    g = snap.readgroup()
    cshids = np.empty((g.size), dtype=np.int64)
    cshids[0]  = 0
    cshids[1:] = np.cumsum(g['nhalo']+1)[:-1]
    s = snap.load('subhalo', 'tab')
    # print s.dtype
    dmpos = snap.load(1, 'pos', s)
    #
    posmin = 0.0
    posmin = 0.0
    posmax = 1.e2 #Mpc/h
    ##
    cshid = cshids[igrp]
    if True :
        cshid = cshid
        x0, y0, z0 = s['pos'][cshid]/1000.0
        #
        rcirc = s['rcirc'][cshid]/1000.0
        npt = dmpos[cshid].shape[0]
        print npt
        if npt > nptmax :
            rid = np.random.randint(0, npt, size=nptmax)
        else :
            rid = np.arange(npt)
        x = dmpos[cshid][rid, 0]/1000.0
        y = dmpos[cshid][rid, 1]/1000.0
        z = dmpos[cshid][rid, 2]/1000.0
        #
        [x, y, z], xyzcen = fix_periodix_box(x, y, z, posmin, posmax,
                xyzcen=[x0, y0, z0])
        x0, y0, z0 = xyzcen
        cow = Cow(x, y, z, xyzcen=None, wt=None) # particles of equal mass
        cow.get_inertia_tensor()
        cow.get_ellipsoid()
        l0, l1, l2 = cow.eigvecs # univectors
        a0, a1, a2 = cow.principal_axes
        v0, v1, v2 = cow.ellipsoid
        cow.get_ellipse(los="x")
        cow.get_ellipse(los="y")
        cow.get_ellipse(los="z")
        xy0, xy1   = cow.ellipse2d["z"]
        eigvecs2d, principal_axes2d, ellipse2d = project_ellipsoid(l0, l1, l2, a0, a1, a2, los="z")
        _xy0, _xy1   = ellipse2d
        yz0, yz1   = cow.ellipse2d["x"]
        eigvecs2d, principal_axes2d, ellipse2d = project_ellipsoid(l0, l1, l2, a0, a1, a2, los="x")
        _yz0, _yz1   = ellipse2d
        zx0, zx1   = cow.ellipse2d["y"]
        eigvecs2d, principal_axes2d, ellipse2d = project_ellipsoid(l0, l1, l2, a0, a1, a2, los="y")
        _zx0, _zx1   = ellipse2d
        # quit()
        if set_plot :
            r = rcirc * 15
            fig, axes = getFrames(3)[:2]
            axes[0].scatter(x, y, marker=".", color="cyan", alpha=0.3)
            axes[0].scatter(x0, y0, marker="*", color="red")
            # axes[0].plot([x0,x0+v0[0]], [y0,y0+v0[1]], color="red"  )
            # axes[0].plot([x0,x0+v1[0]], [y0,y0+v1[1]], color="green")
            # axes[0].plot([x0,x0+v2[0]], [y0,y0+v2[1]], color="blue" )
            axes[0].plot([x0,x0+xy0[0]], [y0,y0+xy0[1]], color="gold"  )
            axes[0].plot([x0,x0+xy1[0]], [y0,y0+xy1[1]], color="gold")
            axes[0].plot([x0,x0+_xy0[0]], [y0,y0+_xy0[1]], color="m", ls="--")
            axes[0].plot([x0,x0+_xy1[0]], [y0,y0+_xy1[1]], color="m", ls="--")
            axes[0].set_xlim(x0-r,x0+r)
            axes[0].set_ylim(y0-r,y0+r)
            axes[1].scatter(y, z, marker=".", color="cyan", alpha=0.3)
            axes[1].scatter(y0, z0, marker="*", color="red")
            # axes[1].plot([y0,y0+v0[1]], [z0,z0+v0[2]], color="red"  )
            # axes[1].plot([y0,y0+v1[1]], [z0,z0+v1[2]], color="green")
            # axes[1].plot([y0,y0+v2[1]], [z0,z0+v2[2]], color="blue" )
            axes[1].plot([y0,y0+yz0[0]], [z0,z0+yz0[1]], color="gold")
            axes[1].plot([y0,y0+yz1[0]], [z0,z0+yz1[1]], color="gold")
            axes[1].plot([y0,y0+_yz0[0]], [z0,z0+_yz0[1]], color="m", ls="--")
            axes[1].plot([y0,y0+_yz1[0]], [z0,z0+_yz1[1]], color="m", ls="--")
            axes[1].set_xlim(y0-r,y0+r)
            axes[1].set_ylim(z0-r,z0+r)
            axes[2].scatter(z, x, marker=".", color="cyan", alpha=0.3)
            axes[2].scatter(z0, x0, marker="*", color="red")
            # axes[2].plot([z0,z0+v0[2]], [x0,x0+v0[0]], color="red"  )
            # axes[2].plot([z0,z0+v1[2]], [x0,x0+v1[0]], color="green")
            # axes[2].plot([z0,z0+v2[2]], [x0,x0+v2[0]], color="blue" )
            axes[2].plot([z0,z0+zx0[0]], [x0,x0+zx0[1]], color="gold")
            axes[2].plot([z0,z0+zx1[0]], [x0,x0+zx1[1]], color="gold")
            axes[2].plot([z0,z0+_zx0[0]], [x0,x0+_zx0[1]], color="m", ls="--")
            axes[2].plot([z0,z0+_zx1[0]], [x0,x0+_zx1[1]], color="m", ls="--")
            axes[2].set_ylim(x0-r,x0+r)
            axes[2].set_xlim(z0-r,z0+r)
            fig.suptitle("central subhalo-"+str(igrp))
            figure_handler(fig, figout="fof"+str(igrp)+"_"+"sh"+str(cshid), figext=figext)

def get_central_subhalo_shape(snapid, ROOT="/physics/yfeng1/mb2/",
        hdfout="csh.h5", mshmin=1.e12, nptmax=1e4):
    """ all halos with position, velocity, mass, etc. not sure whether to
    calculate moment of inertia yet.  """
    snap = SnapDir(snapid, ROOT)
    g = snap.readgroup()
    cshids = np.empty((g.size), dtype=np.int64)
    print "read %10d groups" % cshids.size
    cshids[0]  = 0
    cshids[1:] = np.cumsum(g['nhalo']+1)[:-1]
    s = snap.load('subhalo', 'tab')
    dmpos = snap.load(1, 'pos', s)
    posmin = 0.0
    posmin = 0.0
    posmax = 1.e2 #Mpc/h
    ##
    mshs = s['mass'][cshids] * 1.e10
    mshs = np.nan_to_num(mshs)
    isel = np.where(mshs >= mshmin)[0]
    nsel = isel.size
    print "selected %10d central subhalos " % nsel
    # info needed
    mass = mshs[isel]
    pos  = np.empty((nsel, 3))
    vel  = np.empty((nsel, 3))
    # 3d
    abc  = np.empty((nsel, 3)) # major, intermediate, minor axes
    pa0  = np.empty((nsel, 3)) # univectors
    pa1  = np.empty((nsel, 3))
    pa2  = np.empty((nsel, 3))
    for i, igrp in enumerate(isel) :
        print i
        cshid = cshids[igrp]
        x0, y0, z0 = s['pos'][cshid]/1000.0
        vx, vy, vz = s['vel'][cshid]
        #
        pos[i, :] = x0, y0, z0
        vel[i, :] = vx, vy, vz
        npt = dmpos[cshid].shape[0]
        if npt == 0 :
            print "zero particles?"
            break
        if npt > nptmax :
            rid = np.random.randint(0, npt, size=nptmax)
        else :
            rid = np.arange(npt)
            print npt
        x = dmpos[cshid][rid, 0]/1000.0
        y = dmpos[cshid][rid, 1]/1000.0
        z = dmpos[cshid][rid, 2]/1000.0
        [x, y, z], xyzcen = fix_periodix_box(x, y, z, posmin, posmax,
                xyzcen=[x0, y0, z0])
        x0, y0, z0 = xyzcen
        cow = Cow(x, y, z, xyzcen=None, wt=None) # particles of equal mass
        cow.get_inertia_tensor()
        # 3d
        cow.get_ellipsoid()
        l0, l1, l2 = cow.eigvecs # univectors
        a0, a1, a2 = cow.principal_axes
        v0, v1, v2 = cow.ellipsoid
        # recording
        abc[i, :] = a0, a1, a2 # length along each principal axes, sorted
        pa0[i, :] = l0
        pa1[i, :] = l1
        pa2[i, :] = l2
    # saving to hdf5
    fname = os.path.join(mockdir, hdfout)
    print("saving to %s" % fname)
    f = h5py.File(fname, "w")
    dset = f.create_dataset("mass", data=mass)
    dset = f.create_dataset("pos" , data=pos)
    dset = f.create_dataset("vel" , data=vel)
    dset = f.create_dataset("abc" , data=abc)
    dset = f.create_dataset("pa0" , data=pa0)
    dset = f.create_dataset("pa1" , data=pa1)
    dset = f.create_dataset("pa2" , data=pa2)
    print("saving done.")
    f.close()

def get_all_subhalo_position(snapid, ROOT="/Users/ying/Data/mb2",
        hdfout='sh.h5', mshmin=1.e10):
    """Just subhalos serving as galaxy number density in the GI correlation."""
    snap = SnapDir(snapid, ROOT)
    s = snap.load('subhalo', 'tab')
    print s.dtype
    mshs = s['mass'] * 1.e10
    mshs = np.nan_to_num(mshs)
    isel = np.where(mshs >= mshmin)[0]
    nsel = isel.size
    print "selected %10d subhalos " % nsel
    # info needed
    mass = mshs[isel]
    pos  = s['pos'][nsel, :]
    vel  = s['vel'][nsel, :]
    ##
    fname = os.path.join(mockdir, hdfout)
    print("saving to %s" % fname)
    f = h5py.File(fname, "w")
    dset = f.create_dataset("mass", data=mass)
    dset = f.create_dataset("pos" , data=pos)
    dset = f.create_dataset("vel" , data=vel)
    f.close()

def fix_periodix_box(_x, _y, _z, posmin, posmax, xyzcen=None) :
    pos = [_x, _y, _z]
    gap = (posmax - posmin) * 0.8
    cen = (posmax + posmin) * 0.5
    for i, p in enumerate(pos) :
        pmin, pmax = np.min(p), np.max(p)
        if pmax - pmin > gap :
            p[np.where(p>cen)] -= posmax
            if xyzcen is not None :
                if xyzcen[i] > cen :
                    xyzcen[i] -= posmax
    return(pos, xyzcen)

class Cow(object):
    def __init__(self, x, y, z, xyzcen=None, wt=None) :
        self.x = x
        self.y = y
        self.z = z
        self.pos = [x, y, z]
        self.npt = x.size
        if  wt is None :
            self.wt = np.ones(self.npt)
        self.wt_sum = np.sum(self.wt)
        self.center(xyzcen)
        self.eigvecs2d = {}
        self.principal_axes2d = {}
        self.ellipse2d = {}

    def center(self, xyzcen=None) :
        if xyzcen is None:
            # calculate center of mass
            self.x0 = np.average(self.x, weights=self.wt)
            self.y0 = np.average(self.y, weights=self.wt)
            self.z0 = np.average(self.z, weights=self.wt)
        else :
            self.x0, self.y0, self.z0 = xyzcen
        self._x = self.x - self.x0
        self._y = self.y - self.y0
        self._z = self.z - self.z0
        self._pos = [self._x, self._y, self._z]

    def get_inertia_tensor(self) :
        inertia_tensor = np.zeros((3, 3))
        for i in xrange(3):
            inertia_tensor[i, i] = np.sum(self.wt * self._pos[i]**2)/self.wt_sum
        for i, j in itertools.combinations(xrange(3), 2):
            inertia_tensor[i, j] = np.sum(self.wt * self._pos[i] * self._pos[j])/self.wt_sum
            # has to be symmetric
            inertia_tensor[j, i] = inertia_tensor[i, j]
        self.inertia_tensor = inertia_tensor

    def get_ellipsoid(self) :
        eigvals, eigvecs = np.linalg.eig(self.inertia_tensor)
        # rank from large to small
        seq = np.argsort(eigvals)[::-1]
        self.eigvecs     = eigvecs[seq]
        self.principal_axes = np.sqrt(eigvals[seq])
        self.ellipsoid      = []
        for j in seq :
            self.ellipsoid.append(eigvecs[:,j] * np.sqrt(eigvals[j]))

    def get_ellipse(self, los="z") :
        if los == "z" :
            _ind = np.array([0,1])
        elif los == "y" :
            _ind = np.array([2,0])
        elif los == "x" :
            _ind = np.array([1,2])
        _mgd = np.meshgrid(_ind, _ind)
        _inertia_tensor = self.inertia_tensor[_mgd]
        eigvals, eigvecs = np.linalg.eig(_inertia_tensor)
        seq = np.argsort(eigvals)[::-1]
        self.eigvecs2d[los]  = eigvecs[seq]
        self.principal_axes2d[los] = np.sqrt(eigvals[seq])
        self.ellipse2d[los]  = []
        for j in seq :
            self.ellipse2d[los].append(eigvecs[:,j] * np.sqrt(eigvals[j]))

if __name__ == "__main__":
    snapid = 85
    if True :
        # XXX the central subhalos are having a lot of redundant strucutres
        # that ought to be part of other subhalos. Very screwy.
        get_central_subhalo_shape(snapid, ROOT="/Users/ying/Data/mb2",
                mshmin=1.e11, nptmax=1e4)
    if False :
        get_all_subhalo_position(snapid, ROOT="/Users/ying/Data/mb2",
                mshmin=1.e10)
    if False :
        nex = 10
        for igrp in np.random.randint(0, 10000, nex) :
        # if True :
            # igrp = 100000
            test_shape(snapid, ROOT="/Users/ying/Data/mb2/", igrp=igrp,
                    set_plot=True, figext=None)
