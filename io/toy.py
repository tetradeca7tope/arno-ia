# Last-modified: 01 May 2014 13:55:46


import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
import h5py
import scipy.stats as stats
from rotate_matrix_test import rotation_matrix_numpy, rotation_matrix_weave, \
        get_rotation_matrix
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from itertools import product, combinations
from numpy.linalg import inv, cholesky
from zypy.zyutil import getFrames, figure_handler
from tpcfbox import wii, wp
import multiprocessing
import cPickle as pickle
from javelin.cholesky_utils import cholesky, chosolve_from_tri, chodet_from_tri


# mockdir = "/Users/ying/Data/mb2/align/"
# mockdir = os.path.expanduser("~/Data/mb2/align/")
# mockdir = os.path.expanduser("/home/yingzu/Data/mb2/align/")
mockdir = os.path.expanduser("/home/yingzu/samy/")

# working directory of this script
workdir = os.path.dirname(os.path.realpath(__file__))


""" A simple LRG alighment model, do 2D at this moment.

csh -> LRG : introduce misalignment between the two major axes
 sh -> galalxies: just the position, and mass cut for selecting galaxies.


for II term: intrinsic ellipticity-intrinsic ellipticity correlation
    c_ab:


for GI term: density-intrinsic shear correlation
    w_g+:


"""


class CentralSubHalos(object) :
    """read and prepare the central subhalo catalog for ellipticity
    catalog."""
    def __init__(self, fname=None) :
        if fname is None :
            hdfin = "csh.h5"
            fname = os.path.join(mockdir, hdfin)
        self.f = h5py.File(fname, "r")
        self.load()

    def load(self) :
        self.a    = self.f['abc'][:,0] # major axis
        self.b    = self.f['abc'][:,1] # inter axis
        self.c    = self.f['abc'][:,2] # minor axis
        self.pa0  = self.f['pa0'][:,:] # univector along major axis
        self.pa1  = self.f['pa1'][:,:] # univector along inter axis
        self.pa2  = self.f['pa2'][:,:] # univector along inter axis
        self.x    = self.f['pos'][:,0]
        self.y    = self.f['pos'][:,1]
        self.z    = self.f['pos'][:,2]
        self.vx   = self.f['vel'][:,0]
        self.vy   = self.f['vel'][:,1]
        self.vz   = self.f['vel'][:,2]
        self.mass = self.f["mass"][:]
        self.ncsh = self.mass.size
        print "load %d central subhalos"  % self.ncsh
        if False :
            # just print one for test
            _i = 10
            print " ".join(["[",",".join([format(r, "15.10f") for r in self.pa0[_i,:]]), "]"])
            print " ".join(["[",",".join([format(r, "15.10f") for r in self.pa1[_i,:]]), "]"])
            print " ".join(["[",",".join([format(r, "15.10f") for r in self.pa2[_i,:]]), "]"])

    def exit(self):
        self.f.close()

class IA_test0(object) :
    """central subhalos to central galaxies."""
    def __init__(self, csh):
        self.csh = csh
        self.ncg = self.csh.ncsh
        self.elsds = np.empty(self.ncg, dtype=Ellipsoid)
        self._load_csh()
        self.e1  = np.empty(self.ncg)
        self.e2  = np.empty(self.ncg)
        self.wii_dict ={}

    def _load_csh(self) :
        for i in xrange(self.ncg) :
            self.elsds[i] = Ellipsoid(
                    self.csh.pa0[i,:], self.csh.pa1[i,:], self.csh.pa2[i,:],
                    self.csh.a[i], self.csh.b[i], self.csh.c[i])
        # this shape is for plugging into tpcfbox
        self.pos = np.empty((self.ncg,3))
        self.pos[:,0] = self.csh.x
        self.pos[:,1] = self.csh.y
        self.pos[:,2] = self.csh.z

    def transform(self, mu=30, sigma=30.0) :
        """ model: rotate major axis by N(0, sigma_theta) in 3D"""
        # 0 to 90 use a truncated gaussian
        lower, upper = 0.0, 90.0
        X = stats.truncnorm((lower-mu)/sigma, (upper-mu)/sigma,
                loc=mu, scale=sigma)
        thetas = X.rvs(self.ncg)*np.pi/180.0 # into sterians
        # first rotate the minor axis by some random angle phi btw 0 and pi/2
        phis = np.random.random(self.ncg) * np.pi/2.
        for i in xrange(self.ncg) :
            # first rotate around major axis
            self.elsds[i].rotate_about_major_axis(phis[i])
            # then rotate around minor axis
            self.elsds[i].rotate_about_minor_axis(thetas[i])
        print("halo rotation done")

    def project(self, los="z") :
        for i in xrange(self.ncg) :
            self.elsds[i].project(los=los)
            # populate the array for tpcfbox
            self.e1[i] = self.elsds[i].proj[los]["e1"]
            self.e2[i] = self.elsds[i].proj[los]["e2"]

    def save_shear_catalog(self, fcat) :
        """save the shear catalog into txt file"""
        _x  = self.pos[:,0]
        _y  = self.pos[:,1]
        _z  = self.pos[:,2]
        _e1 = self.e1
        _e2 = self.e2
        np.savetxt(fcat, np.vstack((_x, _y, _z, _e1, _e2)).T)
        print("shear catalog saved in %s" % fcat)

    def correlate_wii(self, rmin, rmax, rcube, zmax, nbin, ncpu=None,
            set_plot=False, figout=None, figext=None) :
        if ncpu is None :
            ncpu = multiprocessing.cpu_count()
        xg1  = self.pos.T
        wgt1 = np.ones(self.ncg)
        ea1  = self.e1
        eb1  = self.e2
        radii,wp2d,waa2d,wbb2d,wab2d,jackerr,jackerraa,jackerrbb,jackerrab =\
                wii.auto2dii(
                        ncpu,rmin,rmax,rcube,zmax,nbin,xg1,wgt1,ea1,eb1
                        )
        self.wii_dict["rp"]     = radii
        self.wii_dict["wp"]     = wp2d
        self.wii_dict["waa"]    = waa2d
        self.wii_dict["wbb"]    = wbb2d
        self.wii_dict["wab"]    = wab2d
        self.wii_dict["errwp"]  = jackerr
        self.wii_dict["errwaa"] = jackerraa
        self.wii_dict["errwbb"] = jackerrbb
        self.wii_dict["errwab"] = jackerrab
        if set_plot :
            fig, axes = getFrames(4, ncol=2, wid=8, xgut=0.1, ygut=0.1)[:2]
            ax = axes[0]
            ax.errorbar(radii, wp2d, yerr=jackerr)
            ax = axes[1]
            ax.errorbar(radii, waa2d, yerr=jackerraa, marker="o")
            ax = axes[2]
            ax.errorbar(radii, wbb2d, yerr=jackerrbb, marker="o")
            ax = axes[3]
            ax.errorbar(radii, wab2d, yerr=jackerrab, marker="o")
            for ax in axes :
                ax.set_xscale("log")
                ax.axhline(0, color="gray", linestyle="dashed")
                ax.set_xlim(radii[0]*0.9, radii[-1]*1.1)
            figure_handler(fig, figout, figext)

def predict_test0(mu=20.0, sigma=30.0,
        rmin=1.0, rmax=50.0, rcube=100.0, zmax=40.0, nbin=20, ncpu=None,
        fcatalog=None):
    csh = CentralSubHalos()
    ia  = IA_test0(csh)
    ia.transform(mu=mu, sigma=sigma)
    ia.project(los="z")
    if fcatalog is not None :
        ia.save_shear_catalog(fcatalog)
    if False :
        for i in np.random.randint(0, ia.ncg, 10):
            ia.elsds[i].show(los="z")
    ia.correlate_wii(rmin, rmax, rcube, zmax, nbin, ncpu=ncpu)
    return(ia.wii_dict)

def genmock_test0(fmockdata="mock0.dat", **predargs):
    fmockdata = os.path.join(workdir, fmockdata)
    wii_dict = predict_test0(**predargs)
    wii_dict["wp_obs"]=wii_dict["wp"]+np.random.normal(scale=wii_dict["errwp"])
    wii_dict["waa_obs"]=wii_dict["waa"]+np.random.normal(scale=wii_dict["errwaa"])
    wii_dict["wbb_obs"]=wii_dict["wbb"]+np.random.normal(scale=wii_dict["errwbb"])
    wii_dict["wab_obs"]=wii_dict["wab"]+np.random.normal(scale=wii_dict["errwab"])
    print("saving mockdata to %s" % fmockdata)
    pickle.dump(wii_dict, open(fmockdata, "wb"))

def loglike_test0(mu, sigma, fmockdata="mock0.dat", ncpu=None):
    fmockdata = os.path.join(workdir, fmockdata)
    wii_dict_obs = pickle.load(open(fmockdata, "r"))
    wii_dict = predict_test0(mu=mu, sigma=sigma, ncpu=ncpu)
    datavec = np.hstack([
        wii_dict_obs["waa_obs"],
        wii_dict_obs["wbb_obs"],
        ])
    predvec = np.hstack([
        wii_dict["waa"],
        wii_dict["wbb"],
        ])
    errvec = np.hstack([
        wii_dict["errwaa"],
        wii_dict["errwbb"],
        ])
    errmat = np.diag(errvec*errvec)
    _loglike = get_loglike_gaussian(datavec,predvec,errmat)
    return(_loglike)

def get_loglike_gaussian(datavec, predvec, errmat):
    # compute loglike
    _C = errmat
    _b = datavec - predvec
    _U, info = cholesky(_C, inplace=False, raiseinfo=False)
    if info > 0 :
        _p = -np.inf
        _q = -np.inf
    else:
        _x = chosolve_from_tri(_U, _b)
        _p = -0.5*np.dot(_b, _x)
        _q = -0.5*chodet_from_tri(_U, retlog=True)
    _log_like = _p + _q
    return(_log_like)

def project_ellipsoid(ss0, ss1, ss2, w0, w1, w2, los="z", return_moi=False):
    """ Joachimi et al. 2013

    ss0 : direction of major axis
    ss1 : direction of inter axis
    ss2 : direction of minor axis
    w0  : major axis
    w1  : inter axis
    w2  : minor axis
    """
    if los == "z" :
        s0 = ss0
        s1 = ss1
        s2 = ss2
    elif los == "y" :
        _seq = np.array([2,0,1])
        s0 = ss0[_seq]
        s1 = ss1[_seq]
        s2 = ss2[_seq]
    elif los == "x" :
        _seq = np.array([1,2,0])
        s0 = ss0[_seq]
        s1 = ss1[_seq]
        s2 = ss2[_seq]
    Sperp = np.empty((2,3)) #xy-plane, eigen-axis
    Sperp[:,0] = s0[:2]
    Sperp[:,1] = s1[:2]
    Sperp[:,2] = s2[:2]
    Spara = np.empty((1,3))
    Spara[0,0] = s0[2]
    Spara[0,1] = s1[2]
    Spara[0,2] = s2[2]
    W = np.array([w0, w1, w2])
    K = np.dot(Spara/W**2, Sperp.T)
    alpha = Spara/W
    alpha2 =np.dot(alpha, alpha.T)
    kka2 = K *  K.T/alpha2
    ssw2 = np.dot(Sperp/W**2, Sperp.T)
    moiinv = ssw2 - kka2
    moi = inv(moiinv)
    if return_moi :
        # e1 and e2
        e1 = (moi[0,0] - moi[1,1])/(moi[0,0] + moi[1,1])
        e2 = (2.*moi[0,1])/(moi[0,0] + moi[1,1])
        ellipticity = [e1, e2]
    eigvals, eigvecs = np.linalg.eig(moi)
    # sort them so that the first index goes to the major axis
    seq = np.argsort(eigvals)[::-1]
    principal_axes2d = np.sqrt(eigvals[seq])
    eigvecs2d = []
    for j in seq :
        eigvecs2d.append(eigvecs[:,j])
    if return_moi :
        return(eigvecs2d, principal_axes2d, moi, ellipticity)
    else :
        return(eigvecs2d, principal_axes2d)

def project_ellipsoid2(moi3d, los="z", return_moi=False):
    """ Joachimi et al. 2013, appendix
    """
    # nvec = np.empty((1,3))
    nvec = np.empty((3,1))
    if los == "z" :
        nvec[:,0] = np.array([0,0,1])
        _seq = np.array([0,1])
    elif los == "y" :
        nvec[:,0] = np.array([0,1,0])
        _seq = np.array([2,0])
    elif los == "x" :
        nvec[:,0] = np.array([1,0,0])
        _seq = np.array([1,2])
    mesh = np.meshgrid(_seq, _seq)
    minv = inv(moi3d)
    rhsup = np.dot(minv, np.dot(nvec,np.dot(nvec.T, minv)))
    rhsbo = np.dot(nvec.T,np.dot(minv, nvec))
    winv = minv - rhsup/rhsbo
    # moi = inv(winv[:2,:2])
    moi = inv(winv[mesh])
    if return_moi :
        e1 = (moi[0,0] - moi[1,1])/(moi[0,0] + moi[1,1])
        e2 = (2.*moi[0,1])/(moi[0,0] + moi[1,1])
        ellipticity = [e1, e2]
    eigvals, eigvecs = np.linalg.eig(moi)
    # sort them so that the first index goes to the major axis
    seq = np.argsort(eigvals)[::-1]
    principal_axes2d = np.sqrt(eigvals[seq])
    eigvecs2d  = []
    for j in seq :
        eigvecs2d.append(eigvecs[:,j])
    if return_moi :
        return(eigvecs2d, principal_axes2d, moi, ellipticity)
    else :
        return(eigvecs2d, principal_axes2d)

class Ellipsoid(object) :
    def __init__(self, pa0, pa1, pa2, a, b, c) :
        # input format of the ellipsoid
        self.pa0 = pa0
        self.pa1 = pa1
        self.pa2 = pa2
        self.a   = a
        self.b   = b
        self.c   = c
        # current format of the ellipsoid
        self._pa0 = pa0
        self._pa1 = pa1
        self._pa2 = pa2
        self._a   = a
        self._b   = b
        self._c   = c
        # get the original
        self._get_moi()
        self.moi = self._moi
        # projection
        self.proj = {}

    def rotate_about_minor_axis(self, theta) :
        """rotate around its minor axis"""
        self.rotate(self._pa2, theta)

    def rotate_about_median_axis(self, theta) :
        """rotate around its median axis"""
        self.rotate(self._pa1, theta)

    def rotate_about_major_axis(self, theta) :
        """rotate around its major axis"""
        self.rotate(self._pa0, theta)

    def rotate(self, axis, theta) :
        """
        rotate the ellipsoid around 'axis' by 'theta'.
        """
        _m = rotation_matrix_numpy(axis, theta)
        self._pa0 = np.dot(_m, self._pa0)
        self._pa1 = np.dot(_m, self._pa1)
        self._pa2 = np.dot(_m, self._pa2)
        self._get_moi()

    def _get_moi(self) :
        """ get the original moment of intertia (covariance matrix)
        using eigenvalues and eigenvectors.
        """
        # http://math.stackexchange.com/a/21801/16618
        P = np.empty((3,3))
        P[:,0] = self._pa0
        P[:,1] = self._pa1
        P[:,2] = self._pa2
        S = np.diag([self._a**2, self._b**2, self._c**2])
        self._moi = np.dot(P, np.dot(S, P.T))

    def show(self, ax=None, plotAxes=False, cageColor="b", cageAlpha=0.2,
            los=None, rescaleAxes=True, figout=None, figext=None) :
        """plot the ellipsoid."""
        make_ax = ax == None
        if make_ax:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_aspect('equal')
        u = np.linspace(0.0, 2.0 * np.pi, 100)
        v = np.linspace(0.0, np.pi, 100)
        # cartesian coordinates that correspond to the spherical angles:
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        # rotate accordingly using cholesky trick
        _rot = np.linalg.cholesky(self._moi)
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i,j],y[i,j],z[i,j]] = np.dot(_rot, [x[i,j],y[i,j],z[i,j]])
        if True :
            if True :
                _pa0 = self._pa0 * self._a
                _pa1 = self._pa1 * self._b
                _pa2 = self._pa2 * self._c
                ax.plot([0, _pa0[0]],[0, _pa0[1] ],[0, _pa0[2]], color="0.2")
                ax.plot([0, _pa1[0]],[0, _pa1[1] ],[0, _pa1[2]], color="0.2")
                ax.plot([0, _pa2[0]],[0, _pa2[1] ],[0, _pa2[2]], color="0.2")
                # make some purdy axes
                axes = [_pa0, _pa1, _pa2]
            # plot axes
            X = np.empty([3,2])
            Y = np.empty([3,2])
            Z = np.empty([3,2])
            for i, p in enumerate(axes):
                X3 = np.linspace(-p[0], p[0], 100)
                Y3 = np.linspace(-p[1], p[1], 100)
                Z3 = np.linspace(-p[2], p[2], 100)
                if plotAxes :
                    ax.plot(X3, Y3, Z3, color=cageColor)
                X[i, :] = np.array([X3.min(), X3.max()])
                Y[i, :] = np.array([Y3.min(), Y3.max()])
                Z[i, :] = np.array([Z3.min(), Z3.max()])
        # plot ellipsoid
        ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=cageColor,
                alpha=cageAlpha)
        if los is not None :
            # plot the projection (currently just z)
            if los == "z" :
                npt = 100
                tt = np.linspace(0, np.pi*2.0, npt)
                xy = np.empty((2,npt))
                _a = self.proj[los]["a"]
                _b = self.proj[los]["b"]
                xy[0, :] = np.cos(tt)
                xy[1, :] = np.sin(tt)
                L = cholesky(self.proj[los]["moi"])
                _xy = np.dot(L, xy)
                _z = np.zeros(npt)
                ax.plot(_xy[0,:], _xy[1,:], _z, color="r")
                _pa0 = self.proj[los]["pa0"] * _a
                _pa1 = self.proj[los]["pa1"] * _b
                ax.plot([0, _pa0[0]],[0, _pa0[1] ],[0, 0], color="r")
                ax.plot([0, _pa1[0]],[0, _pa1[1] ],[0, 0], color="r")
                # view along Z-axis
                ax.view_init(90, 0)
        if rescaleAxes :
            sca = 2.0
            max_range = sca * np.array([
                X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()/2.0
            mean_x = X.mean()
            mean_y = Y.mean()
            mean_z = Z.mean()
            ax.set_xlim(mean_x - max_range, mean_x + max_range)
            ax.set_ylim(mean_y - max_range, mean_y + max_range)
            ax.set_zlim(mean_z - max_range, mean_z + max_range)
        if make_ax:
            figure_handler(fig, figout, figext)
            plt.close(fig)
            del fig

    def project(self, los="z") :
        """how to project this mother fucker."""
        if True :
            # presumably faster
            eigvecs2d, principal_axes2d, moi2d, ellipticity = project_ellipsoid(
                self._pa0,self._pa1,self._pa2, self._a, self._b, self._c,
                los=los ,return_moi=True)
        else :
            eigvecs2d, principal_axes2d, moi2d, ellipticity = project_ellipsoid2(
                self._moi, los=los ,return_moi=True)
        self.proj[los] = {}
        self.proj[los]["pa0"] = eigvecs2d[0]
        self.proj[los]["pa1"] = eigvecs2d[1]
        self.proj[los]["a"]   = principal_axes2d[0]
        self.proj[los]["b"]   = principal_axes2d[1]
        self.proj[los]["moi"] = moi2d
        self.proj[los]["e1"]  = ellipticity[0]
        self.proj[los]["e2"]  = ellipticity[1]

def test_ellipticity():
    """ Test different ways of calculating ellipticity.
    """
    if False :
        x = 0.3*np.pi
        pa0 = np.array([ np.cos(x), -np.sin(x), 0])
        pa1 = np.array([ np.sin(x), +np.cos(x), 0])
        # no matter how you cross the two, results don't matter
        # pa2 = np.cross(pa1, pa0)
        pa2 = np.cross(pa0, pa1)
    if False :
        pa0 =  np.array([ 0.73325756, 0.66042404, -0.16178205])
        pa1 =  np.array([ 0.02967341, 0.20662449,  0.97797025])
        pa2 =  np.array([-0.6793032 , 0.72190471, -0.13191194])
    if True :
        pa0 =  np.array([ -0.5010498010,  -0.7133443206,   0.4899887521 ])
        pa1 =  np.array([ -0.8620956496,   0.3618504848,  -0.3547609301 ])
        pa2 =  np.array([ -0.0757640272,   0.6001700650,   0.7962761489 ])
    #
    a,b,c   = 1.0, 0.5, 0.3
    eigvecs2d, principal_axes2d = project_ellipsoid(
            pa0,pa1,pa2, a, b, c, los="z")
    q = principal_axes2d[1]/principal_axes2d[0]
    # XXX using +x as positive e1 axis
    beta = np.arctan(eigvecs2d[0][1]/eigvecs2d[0][0]) # between -pi/2 and pi/2
    _beta = beta *180.0/np.pi
    print "position angle: %4.2f" %  _beta
    r = (1.-q*q)/(1.+q*q)
    print "\n"
    print "From Inverting 2D MoI Matrix"
    print np.cos(2.*beta)*r
    print np.sin(2.*beta)*r
    # cosbeta = eigvecs2d[0][0]
    # sinbeta = eigvecs2d[0][1]
    # print (cosbeta*cosbeta-sinbeta*sinbeta)*(1.-q*q)/(1.+q*q)
    # print (2.*cosbeta*sinbeta)*(1.-q*q)/(1.+q*q)
    if False :
        fig, axes = getFrames(1, wid=8)[:2]
        ax = axes[0]
        ax.plot([0., pa0[0]*a], [0.0, pa0[1]*a], "k-")
        ax.plot([0., pa1[0]*b], [0.0, pa1[1]*b], "k-")
        _pa0 = eigvecs2d[0]
        _pa1 = eigvecs2d[1]
        _a = principal_axes2d[0]
        _b = principal_axes2d[1]
        ax.plot([0., _pa0[0]*_a], [0.0, _pa0[1]*_a], "g--")
        ax.plot([0., _pa1[0]*_b], [0.0, _pa1[1]*_b], "g--")
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        figure_handler(fig)

def test_Ellipsoid(figout=None, figext=None):
    """visualize the ellipsoid object"""
    if False :
        pa0 =  np.array([ -0.5010498010,  -0.7133443206,   0.4899887521 ])
        pa1 =  np.array([ -0.8620956496,   0.3618504848,  -0.3547609301 ])
        pa2 =  np.array([ -0.0757640272,   0.6001700650,   0.7962761489 ])
    if True :
        pa0 =  np.array([ 1.        , 0.        ,  0.        ])
        pa1 =  np.array([ 0.        , 1.        ,  0.        ])
        pa2 =  np.array([ 0.        , 0.        ,  1.        ])
    if False :
        pa0 =  np.array([ 0.8       ,-0.6       ,  0.        ])
        pa1 =  np.array([ 0.6       , 0.8       ,  0.        ])
        pa2 =  np.array([ 0.        , 0.        ,  1.        ])
    a, b, c = 6.0, 3.0, 2.0
    elsd = Ellipsoid(pa0, pa1, pa2, a, b, c)
    axis = np.array([0.0, 0.0, 1.0])
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_aspect('equal')
    elsd.show(ax=ax)
    elsd.rotate(axis, np.pi*0.5)
    elsd.project(los="z")
    elsd.show(ax=ax,los="z")
    figure_handler(fig, figout=None, figext=None)

def test_genmock() :
    """ generate mocks."""
    genmock_test0(fmockdata="mock0.dat", mu=20.0, sigma=30.0)

def test_codecomparison() :
    """ generate shear catalog for code comparison"""
    fcatalog = "mock0_shearcat.txt"
    if False :
        genmock_test0(fmockdata="mock0_codecomp.dat", mu=20.0, sigma=30.0,
                rmin=1.0, rmax=50.0, rcube=100.0, zmax=40.0, nbin=20,
                ncpu=None, fcatalog=fcatalog)
    if True :
        x, y, z, ea1, eb1 = np.genfromtxt(fcatalog, unpack=True)
        xg1 = np.empty((3, x.size))
        wgt1= np.ones(x.size)
        xg1[0,:] = x
        xg1[1,:] = y
        xg1[2,:] = z
        rmin=1.0
        rmax=50.0
        rcube=100.0
        zmax=40.0
        nbin=20
        ncpu=8
        if True :
            radii, wp2d, jackerr = wp.auto2d(
                            ncpu,rmin,rmax,rcube,zmax,nbin,xg1,wgt1
                    )
            result = np.vstack((radii,wp2d,jackerr)).T
            np.savetxt('codecomp/output_yz_wp.dat', result)
        else :
            radii,wp2d,waa2d,wbb2d,wab2d,jackerr,jackerraa,jackerrbb,jackerrab =\
                    wii.auto2dii(
                            ncpu,rmin,rmax,rcube,zmax,nbin,xg1,wgt1,ea1,eb1
                            )
            result = np.vstack((radii,wp2d,waa2d,wbb2d,wab2d,
                jackerr,jackerraa,jackerrbb,jackerrab)).T
            np.savetxt('codecomp/output_yz.dat', result)

def test_loglike() :
    mu = 20.0
    sigma = 30.0
    print loglike_test0(mu, sigma, ncpu=8)

if __name__ == "__main__":
    # test_codecomparison()
    if True :
        import sys
        inFile = sys.argv[1];
        outFile = sys.argv[2];
        fin = open(inFile, 'r');
        inArgs = fin.read().strip();
        inArgs = inArgs.split();
        x = [float(elem) for elem in inArgs];

        # Now write to a file
        fout = open(outFile, 'w');
        mu, sigma = x
        result = str(loglike_test0(mu=mu, sigma=sigma));
        fout.write(result);

        # close files
        fin.close();
        fout.close();



