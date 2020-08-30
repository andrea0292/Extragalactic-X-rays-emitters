import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.misc import derivative
import astropy.units as u
import astropy.constants as c
from astropy.cosmology import FlatLambdaCDM, z_at_value
from tqdm import *
from astropy.cosmology import Planck13 as cosmo
from astropy import constants as const
import sys
from nbodykit.lab import *
from nbodykit import setup_logging, style

import matplotlib.pyplot as plt
from scipy.optimize import fmin as simplex

gasaverage2 = np.loadtxt('/Users/andreacaputo/Desktop/Phd/LineIntensityMapping/NewPower/rho2gas_ave.dat')

tointta= interp1d(gasaverage2[:,0],gasaverage2[:,1]) # to interpolate



def fungas(z):

    if z < 0.01:
            
        return tointta(0.01)
    
    return 1.*tointta(z)

class Xray_signal:
    def __init__(self):
        """
        Container class to compute the X ray diffuse signal from AGN, Galaxies and Clusters of Galaxies
        """

        # Set constants

        self.Mpctocm = 3.086e24 # from Mpc to cm
        self.fromergtokev = 624150647.99632 # from erg to keV
        self.mtoev = 1/(1.97 * 1e-7) 
        self.zc = 0.75
        self.p1 = 6.36
        self.p2 = -0.24
        self.L0 = 10**44.77 * self.fromergtokev # keV/s
        self.gamma1 = 0.62
        self.gamma2 = 3.01

        # AGN parameters, take a look at https://arxiv.org/pdf/1911.09120.pdf, https://arxiv.org/pdf/1505.07829.pdf

        self.Emin = 2 # keV
        self.Emax = 8 # keV
        self.Etildemin = 0.5 # keV
        self.Etildemax = 2 # keV

        self.Fsens = 2.4e-17 * self.fromergtokev #3e-13 * fromergtokev  keV/s/cm**2 see --> https://arxiv.org/abs/1210.0550
        self.Gamma = 1.45 # spectral index
        self.Lmin = 1e41 * self.fromergtokev # keV/s from https://arxiv.org/pdf/1505.07829.pdf above Eq.2.9
        
        # Parameters for the galaxies https://arxiv.org/pdf/1911.09120.pdf, https://arxiv.org/pdf/1505.07829.pdf

        self.alfa = 1.43
        self.phistar = 10**(-2.23) # Mpc^-3
        self.sigma = 0.72

        self.EmaxG = 2
        self.EminG = 0.5

        self.LXminGal = 1e38*self.fromergtokev

        # Parameters for the clusters https://arxiv.org/pdf/1911.09120.pdf, https://arxiv.org/pdf/1505.07829.pdf

        self.rhocr = 8.55461*10**(-27) # Kg/m^3
        self.mproton = const.m_p.value # Kg
        self.Omegab = cosmo.Ob0 
        self.T0gas = 1 # keV 
        self.T0gasK = self.T0gas*1e3/8.617e-5 #in Kelvin

    # Now we compute the window functions for the three contributions. First we start wit the AGN

    def Kk(self, z):
        return 10**(-4.53 - 0.19*(1+z)) # Mpc**-3

    def Lstar(self, z):
        return self.L0 * ( ((1+self.zc)/(1+z))**self.p1 + ((1+self.zc)/(1+z))**self.p2  )**(-1)

    def phi(self, L,z):
    
        return self.Kk(z) * ( (L/self.Lstar(z))**self.gamma1 + (L/self.Lstar(z))**self.gamma2  ) **-1 # Mpc**-3

    def dFdz(self, E):
    
        return  (2 - self.Gamma) / (self.Emax**(2 - self.Gamma) - self.Emin**(2 - self.Gamma)) * E**(-self.Gamma)

    # The maximum luminosity as dimensions erg/s; Fsens * distL**2 = erg/s/cm^2 * cm^2 = erg/s

    def Lmax(self, z):

        distL = cosmo.luminosity_distance(z).value * self.Mpctocm # in cm
    
        factor = (self.Emax**(2 - self.Gamma) - self.Emin**(2 - self.Gamma)) / (self.Etildemax**(2 - self.Gamma) - self.Etildemin**(2 - self.Gamma))
    
        return 4*np.pi*distL**2 * self.Fsens* 1/((1 + z)**(2 - self.Gamma)) * factor 

    # The energy needs to be passed in keV, the result will be in 1/keV/s/Mpc**3/sr

    def WindAGN(self, E,z):

        MaxL = self.Lmax(z)
        L_ary = np.linspace(self.Lmin, MaxL,50)
        to_int_ary = np.array([self.dFdz(E)*self.phi(L,z) for L in L_ary])
    
        return np.trapz(to_int_ary, L_ary)/ 4/np.pi

    # Now we pass to galaxies window function

    def LmaxGal(self, z):
        
        distL = cosmo.luminosity_distance(z).value * self.Mpctocm
    
        return 4*np.pi*distL**2 * self.Fsens # 1/keV/s

    def phiGal(self, Lx, z):
    
        def Lstar(z):
        
            return 10**39.74 * ((1+z)/1.25)**1.9 * self.fromergtokev # keV/s
    
        return  self.phistar * (Lx / self.Lstar(z))**(1-self.alfa) * np.exp(-1/2/self.sigma**2 * (np.log10(1+Lx / self.Lstar(z)))**2)


    def dFdzGal(self, E):
    
        return 1/E**2 / np.log(self.EmaxG/self.EminG)

    def WindGal(self, E,z):
    
        MaxL = self.LmaxGal(z)
        L_ary = np.linspace(self.LXminGal, MaxL,50)
        to_int_ary = np.array([self.dFdzGal(E)*self.phiGal(L, z) for L in L_ary])
    
        return np.trapz(to_int_ary, L_ary) / 4/np.pi # 1/keV/s/Mpc**3/sr

    #Now we pass to the cluster contribution; <rho_gas^2> is an adimensional quantity. This contribution comes from 
    # free-free (thermal brem) from the gas. Notice that respect to dark matter decay this contributions scales with the desnity squared


    def WindCluster(self, E,z):

        gff = 1.1 #  Gaunt factor for the free-free emission
        kff = 1.03*10**(-17)*gff;
    
        return kff * (self.Omegab * self.rhocr/self.mproton)**2/(4*np.pi)/np.sqrt(self.T0gasK)/(E*(1+z))* np.exp(-E*(1+z)/self.T0gas) * fungas(z) # s-1 keV-1 m-3 sr-1

