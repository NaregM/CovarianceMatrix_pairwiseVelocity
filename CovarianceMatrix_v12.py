import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

import scipy as sp
from scipy import special
from scipy.integrate import quad

from matplotlib import cm

import camb
from camb import model, initialpower

from colossus.cosmology import cosmology
from colossus.halo import mass_so, mass_defs
from colossus.lss import bias
from colossus.lss import mass_function

# ========================================================================================================================================
cosmology.setCosmology('planck15');   # This can be more granular

# ========================================================================================================================================
# CovarianceMatrix_v12 calculates cosmic variance and shot noise for pairwise velocity
class CovarianceMatrix_v12():


    def __init__(self, z, M, fsky, r_min, r_max, Dr, h = 0.67, ombh2 = 0.02256, omch2 = 0.1142,
                 ns = 0.97):

        self.M = M
        self.z = z
        self.b = bias.haloBias(M, model = 'tinker10', z = z, mdef = 'vir')

        self.PS = None

        self.r_min = r_min
        self.r_max = r_max
        self.Dr = Dr

        self.h = h
        self.ombh2 = ombh2
        self.omch2 = omch2
        self.ns = ns

        self.fsky = fsky


    def make_grid(self):

        R = np.arange(self.r_min, self.r_max, self.Dr)

        return np.meshgrid(R, R, indexing = 'ij')


    def setPower(self):

        pars = camb.CAMBparams()
        pars.set_cosmology(H0 = self.h * 100, ombh2 = self.ombh2, omch2 = self.omch2)
        pars.set_dark_energy()
        pars.InitPower.set_params(ns = self.ns)

        pars.Nonlinear = model.NonLinear_none
        results = camb.get_results(pars)

        PK = camb.get_matter_power_interpolator(pars, nonlinear = False, hubble_units = True,
                                                k_hunit = True, kmax = 5.0)

        self.PS = PK

        print("Power is set, you can start the calculation!")


    def E(self, Om0 = 0.2815, Omegak = 0.0, w0 = -1, w1 = 0):

        z = self.z
        a = 1.0/(1+z)

        return np.sqrt(Om0 * a**(-3) + (1 - Om0) * a**(-3 * (1 + w0 + w1)) * np.exp(3 * w1 * (a - 1)) + Omegak * a**(-2))


    def Omegam(self, Om0 = 0.2815):

        z = self.z

        return Om0 * (1 + z)**3 / self.E()**2


    def Omegade(self, Om0 = 0.2815, w0 = -1, w1 = 0):

        z = self.z
        Ode = 1 - Om0

        return Ode * (1 + z)**(3 * (1.0 + w0 + w1)) * np.exp(-3.0 * w1 * z/(1 + z))/(self.E())**2


    def Xi(self, Om0 = 0.2815):

        """
        Comoving radial distance
        """
        z = self.z

        return 3e8/1.0e5 * quad(lambda z: 1/self.E(), 0, z)[0]


    def DA(self, Om0 = 0.2815):

        """
        Angular diameter distance h^-1Mpc
        """
        z = self.z

        return (1 + z)**(-1) * self.Xi()


    def W(self, x):

        return (2*np.cos(x) + x * np.sin(x))/x**3


    def W_Delta(self, k, r, Dr = 2):

        Rmin = r - Dr
        Rmax = r + Dr

        return 3 * (Rmin**3 * self.W(k*Rmin) - Rmax**3 * self.W(k*Rmax))/(Rmax**3 - Rmin**3)


    def z2a(self, z):

        """
        Redshift to scale factor
        """

        return 1./(1 + z)


    def H(self):

        return H0 * self.E()


    def Omega_m(self):

        Omega_m0 = 0.2815

        return Omega_m0 * (1 + self.z)**3 / (Omega_m0 * (1. + self.z)**3 + 1 - Omega_m0)


    def CmovingVolume(self):

        return 4.0 * np.pi/3 * self.Xi()**3


    def xi_quad(self, r):

        z = self.z
        PK_ = self.PS

        integrand = lambda k: 1.0/(2*np.pi**2 * r) * k * np.sin(k*r) * PK_.P(z, k)

        return sp.integrate.quad(integrand , 1e-4, 1, limit= 300)[0]


    def pre_xi(self, r, rp):

        z = self.z

        b = self.b

        return 1./(1 + b**2 * self.xi_quad(r, z)) * (1./(1. + b**2 * self.xi_quad(rp, z)))


    def cosmic_var_func(self, r, rp):

        fsky = self.fsky
        z = self.z
        b = self.b
        n_a =  mass_function.massFunction(self.M, z, mdef = 'fof', model = 'jenkins01', q_out = 'dndlnM')

        PK_ = self.PS

        p0 = (4.0/(np.pi**2 * self.CmovingVolume() * fsky)) * (self.Omegam()**0.55)**2 * (100 * self.h)**2 * self.z2a(z)**2
        p1 = 1./(1 + b**2 * self.xi_quad(r)) * (1./(1. + b**2 * self.xi_quad(rp)))

        return p0 * p1 * quad(lambda k: (2*b*PK_.P(z,k)/n_a + b**2 * PK_.P(z, k)**2) * self.W_Delta(k, r) * self.W_Delta(k, rp), 1e-4, 10, limit = 300)[0]


    def cosmic_var(self):

        if self.PS == None:

            print("Set the power first!")

        r, rp = self.make_grid()

        return np.vectorize(self.cosmic_var_func)(r, rp)


    def shot_noise_func(self, r, rp):


        fsky = self.fsky
        b = self.b
        n_a = mass_function.massFunction(self.M, self.z, mdef = 'fof', model = 'jenkins01', q_out = 'dndlnM')

        p0 = (4.0/(np.pi**2 * self.CmovingVolume() * fsky)) * (self.Omega_m()**0.55)**2 \
                * (100 * self.h)**2 * self.z2a(self.z)**2 * 1./n_a**2

        p1 = 1./(1 + b**2 * self.xi_quad(r)) * (1./(1. + b**2 * self.xi_quad(rp)))

        return p0 * p1 * quad(lambda k: self.W_Delta(k, r) * self.W_Delta(k, rp), 1e-4, 10, limit = 300)[0]


    def shot_noise(self):

        r, rp = self.make_grid()

        return np.vectorize(self.shot_noise_func)(r, rp)
