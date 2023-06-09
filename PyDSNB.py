import configparser

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
import scipy.integrate as integrate
from scipy.integrate import simps
from scipy.interpolate import interp1d
from snewpy.flavor_transformation import AdiabaticMSW
from snewpy.models import ccsn
from snewpy.neutrino import Flavor, MassHierarchy, MixingParameters
from scipy.special import gamma



class units_and_constants:
    """
    Define units and constants used throughout
    """
    
    
    def __init__(self):
        self.MeV    = 1.                                    # Unit of energy: MeV
        self.eV     = 1.e-6*self.MeV
        self.TeV    = 1.e6*self.MeV
        self.erg    = self.TeV/1.602                        # erg
        self.J      = 1.0e7*self.erg                        # joule

        self.cm     = 5.0678e4/self.eV                      # centi-meter
        self.m      = 1.0e2*self.cm
        self.km     = 1.0e3*self.m

        self.pc     = 3.086e18*self.cm                      # persec
        self.kpc    = 1.0e3*self.pc
        self.Mpc    = 1.0e3*self.kpc
        self.s      = 2.9979e10*self.cm                     # second
        self.hour   = 3600.0*self.s
        self.day    = 24.0*self.hour
        self.yr     = 365.2422*self.day

        self.kg     = self.J/self.m**2*self.s**2
        self.gram   = self.kg/1000.
        self.Msun   = 1.989e30*self.kg                      # Mass of the Sun
        self.G      = 6.674e-11*self.m**3/self.kg/self.s**2 # Gravitational constant
        self.deg    = np.pi/180.0                           # Degree
        self.arcmin = self.deg/60.                          # Arcminute
        self.arcsec = self.arcmin/60.                       # Arcsecond
        self.sr     = 1.                                    # Steradian

        
        
        
class cosmology(units_and_constants):
    """
    Define cosmological values
    """
    
    
    def __init__(self):
        units_and_constants.__init__(self)
        self.OmegaB        = 0.049
        self.OmegaM        = 0.315
        self.OmegaC        = self.OmegaM-self.OmegaB
        self.OmegaL        = 1.-self.OmegaM
        self.h             = 0.674
        self.H0            = self.h*100.*self.km/self.s/self.Mpc 
        self.rhocrit0      = 3*self.H0**2/(8.0*np.pi*self.G)
        
    def Hubble(self, z):
        return self.H0*np.sqrt(self.OmegaM*(1.+z)**3+self.OmegaL)
    
    
    
    
class supernova_rate(units_and_constants):
    """
    Definitions related to the rate of supernovae
    """
    
    
    def __init__(self, inifile):
        units_and_constants.__init__(self)
        self.config = configparser.ConfigParser()
        self.config.read(inifile)
        model = self.config['IMF']['model']
        if model=='Salpeter':
            self.xi1 = 2.35
            self.xi2 = 2.35
        elif model=='Kroupa':
            self.xi1 = 1.3
            self.xi2 = 2.3
        elif model=='BG':
            self.xi1 = 1.5
            self.xi2 = 2.15
        
        
    def R_SF(self, z):
        """ 
        Yuksel et al. arXiv:0804.4008
        Horiuchi et al. arXiv:0812.3157  
        """
        R0  = 0.0178*self.Msun/self.yr/self.Mpc**3
        a   = 3.4
        b   = -0.3
        c   = -3.5
        eta = -10.
        z1  = 1.
        z2  = 4.
        B   = (1.+z1)**(1.-a/b)
        C   = (1.+z1)**((b-a)/c)*(1.+z2)**(1.-b/c)
        R_SF = R0*((1.+z)**(a*eta)+((1.+z)/B)**(b*eta) \
                   +((1.+z)/C)**(c*eta))**(1./eta)
        # Conversion factor - this fit assumes Salpeter IMF (from Horiuchi et al. arXiv:0812.3157)
        if self.config['IMF']['model'] == 'Salpeter':
            R_SF = R_SF * 1
        elif self.config['IMF']['model'] == 'Kroupa':
            R_SF = R_SF * 0.66
        elif self.config['IMF']['model'] == 'BG':
            R_SF = R_SF * 0.55
        
        return R_SF

    
    def IMF(self, M):
        """
        model: 'Salpeter' (default), 'Kroupa', or 'BG'
        """
        phi = np.where(M>0.5*self.Msun,(M/(0.5*self.Msun))**-self.xi2,
                       (M/(0.5*self.Msun))**-self.xi1)
        return phi
    
    
    def R_SN(self, z):
        """
        Calculate supernova rate as a function of redshift
        """
        M1 = np.logspace(-1,2,1000)*self.Msun
        M2 = np.logspace(np.log10(8.),2,1000)*self.Msun
        denominator = simps(M1**2*self.IMF(M1),x=np.log(M1))
        numerator   = simps(M2*self.IMF(M2),x=np.log(M2))
        conversion  = numerator/denominator
        R_SN = conversion*self.R_SF(z)
        return R_SN
    
        

        
class SN_neutrino_spectrum(supernova_rate):
    """
    Calculate neutrino spectra depending on preferences defined in the initialization file
    """
    
    
    def __init__(self, inifile):
        supernova_rate.__init__(self, inifile)
        self.flavor = Flavor[self.config['NEUTRINO']['flavor']]
        self.mh = MassHierarchy[self.config['NEUTRINO']['mh'].upper()]
        self.method = self.config['LATEPHASE']['method']
        
        self.SNmodel = self.config['SUPERNOVA']['authors']
        E = np.linspace(0.,100.,2001)*self.MeV
        # In case where spectrum is weighed by the chosen IMF
        if self.config['IMF']['weighed_dNdE']=='T':
            dNdE_calc = self.dNdE_calc_IMFweighed
        # In case spectrum is not IMF weighed and using Nakazato_2013 failed SN model
        elif self.config['IMF']['weighed_dNdE']=='F' and self.config['FAILEDSNE']['failed_SNe']=='T':
            dNdE_calc = self.dNdE_calc_BHweighed
        # In case spectrum is not IMF weighed and using successeful SN model
        elif self.config['IMF']['weighed_dNdE']=='F':
            dNdE_calc = self.dNdE_calc_IMFunweighed
        self.dNdE_func = interp1d(E,dNdE_calc(E),
                                  bounds_error=False,fill_value=0.)
        
    def energy_to_temp(self, en, alpha):
        """ 
        Hudepohl 2014: https://mediatum.ub.tum.de/doc/1177481/1177481.pdf
        Thermal spectrum is alpha ~ 2.3
        """
        return (1/3.597)*1.141*((1/1.30291)*((2+alpha)/(1+alpha)))**(1/2)*en
    
    def find_index(self, lst, number):
        """ 
        Get index in list that matches number - for finding index to stop early phase integration. Works unless data does not go until shock revival time.
        """
        for i in lst:
            if i//number == 1:
                return lst.index(i)
    
    def pinched_FD(self, mean, etot, alpha, energies):
        """ 
        Pinched Fermi Dirac distribution function
        """
        return ((1+alpha)**(1+alpha)/gamma(1+alpha))*((etot*energies**alpha)/(mean**(2+alpha)))*np.exp(-(1+alpha)*energies/mean)
        
        
    def dNdE_calc_IMFunweighed(self, E):
        """ 
        Only implemented for Fornax_2021 and Nakazato_2013 since these revive times and final masses are known.
        """
        if self.config['LATEPHASE']['late_dNdE']=='T':
            F21initmasses = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 26.99]
            F21revtimes = [0, 409.4, 429.9, 0, 255.6, 347.0, 272.8, 309.5, 332.6, 344.2, 365.1, 389.8, 334.6, 315.2, 338.9]
            F21finalmasses = [0, 1.88, 1.80, 0, 1.74, 2.04, 1.79, 1.86, 2.09, 2.25, 2.05, 2.03, 2.10, 2.14, 2.11]
            
            N13inits = [{13, 0.02, 100}, {13, 0.02, 200}, {13, 0.02, 300}, {20, 0.02, 100}, {20, 0.02, 200}, {20, 0.02, 300}, {30, 0.02, 100}, {30, 0.02, 200}, {30, 0.02, 300}, {50, 0.02, 100}, {50, 0.02, 200}, {50, 0.02, 300}, {13, 0.004, 100}, {13, 0.004, 200}, {13, 0.004, 300}, {20, 0.004, 100}, {20, 0.004, 200}, {20, 0.004, 300}, {50, 0.004, 100}, {50, 0.004, 200}, {50, 0.004, 300}]
            N13finalmasses = [1.50, 1.59, 1.64, 1.47, 1.54, 1.57, 1.62, 1.83, 1.98, 1.67, 1.79, 1.87, 1.50, 1.58, 1.63, 1.63, 1.73, 1.77, 1.67, 1.79, 1.91]
            
            if self.SNmodel=='Fornax_2021':
                model = ccsn.Fornax_2021(progenitor_mass=float(self.config['SUPERNOVA']['mass'])*u.solMass)
                revtime = F21revtimes[F21initmasses.index(float(self.config['SUPERNOVA']['mass']))]
                finalmass = F21finalmasses[F21initmasses.index(float(self.config['SUPERNOVA']['mass']))]
            elif self.SNmodel=='Nakazato_2013':
                model = ccsn.Nakazato_2013(progenitor_mass=float(self.config['SUPERNOVA']['mass'])*u.solMass,revival_time=float(self.config['SUPERNOVA']['rev_time'])*u.ms,metallicity=float(self.config['SUPERNOVA']['metallicity']),eos=self.config['SUPERNOVA']['eos'])
                revtime = float(self.config['SUPERNOVA']['rev_time'])
                finalmass = N13finalmasses[N13inits.index({float(self.config['SUPERNOVA']['mass']),float(self.config['SUPERNOVA']['metallicity']),float(self.config['SUPERNOVA']['rev_time'])})]
            else:
                raise Exception('Only implemented for Fornax 2021 and Nakazato 2013 models')
                
            xform = AdiabaticMSW(mh=self.mh)

            # Picks out everything up to revival time (t400)
            t_list = model.time.value[:self.find_index(list(model.time.value),revtime/1000)]*self.s
            N_E = len(E)
            N_t = len(t_list)
            dNdtdE = np.empty((N_t,N_E))
            for i in np.arange(N_t):
                t = t_list[i]
                dNdtdE[i] = model.get_transformed_spectra((t/self.s)*u.s,
                                                          (E/self.MeV)*u.MeV,
                                                          xform)[self.flavor].value \
                    *self.erg**-1*self.s**-1
            dNdE = simps(dNdtdE,x=t_list,axis=0)
            
            Etot = (self.lateliberatedenergy(finalmass,revtime,Flavor.NU_E)*self.erg)+\
                (self.lateliberatedenergy(finalmass,revtime,Flavor.NU_E_BAR)*self.erg)+\
                4*(self.lateliberatedenergy(finalmass,revtime,Flavor.NU_X)*self.erg)
            Tnuebar = self.energy_to_temp(self.latemeanenergy(finalmass,revtime,Flavor.NU_E_BAR),2.3)*self.MeV
            Tnux = self.energy_to_temp(self.latemeanenergy(finalmass,revtime,Flavor.NU_X),2.3)*self.MeV

            dNdE2_nuebar = Etot/6.*120./(7.*np.pi**4)*E**2/Tnuebar**4 \
                /(np.exp(E/Tnuebar)+1.)
            dNdE2_nux = Etot/6.*120./(7.*np.pi**4)*E**2/Tnux**4 \
                /(np.exp(E/Tnux)+1.)
            
            theta12,theta13,theta23 = MixingParameters(mh=self.mh).get_mixing_angles()
            theta12 = theta12.value*self.deg
            
            if self.mh==MassHierarchy.NORMAL:
                dNdE2 = np.cos(theta12)**2*dNdE2_nuebar+np.sin(theta12)**2*dNdE2_nux
            if self.mh==MassHierarchy.INVERTED:
                dNdE2 = np.sin(theta12)**2*dNdE2_nuebar+np.cos(theta12)**2*dNdE2_nux
        
            return dNdE + dNdE2
        else:
            if self.SNmodel=='FermiDirac':
                Etot = float(self.config['SUPERNOVA']['Etot_erg'])*self.erg
                if self.flavor==Flavor.NU_E_BAR:
                    Tnuebar = float(self.config['SUPERNOVA']['Tnuebar_MeV'])*self.MeV
                    Tnux = float(self.config['SUPERNOVA']['Tnux_MeV'])*self.MeV

                dNdE_nuebar = Etot/6.*120./(7.*np.pi**4)*E**2/Tnuebar**4 \
                    /(np.exp(E/Tnuebar)+1.)
                dNdE_nux = Etot/6.*120./(7.*np.pi**4)*E**2/Tnux**4 \
                    /(np.exp(E/Tnux)+1.)
                
                theta12,theta13,theta23 = MixingParameters(mh=self.mh).get_mixing_angles()
                theta12 = theta12.value*self.deg
                
                if self.mh==MassHierarchy.NORMAL:
                    dNdE = np.cos(theta12)**2*dNdE_nuebar+np.sin(theta12)**2*dNdE_nux
                if self.mh==MassHierarchy.INVERTED:
                    dNdE = np.sin(theta12)**2*dNdE_nuebar+np.cos(theta12)**2*dNdE_nux
            elif self.SNmodel=='PinchedFermiDirac':
                Etot = float(self.config['SUPERNOVA']['Etot_erg'])*self.erg
                Enuebar = float(self.config['SUPERNOVA']['Enuebar_MeV'])*self.MeV
                Enux = float(self.config['SUPERNOVA']['Enux_MeV'])*self.MeV
                Alpha = float(self.config['SUPERNOVA']['alpha'])
                
                dNdE_nuebar = self.pinched_FD(Enuebar, Etot/6, Alpha, E)
                dNdE_nux = self.pinched_FD(Enux, Etot/6, Alpha, E)
                
                theta12,theta13,theta23 = MixingParameters(mh=self.mh).get_mixing_angles()
                theta12 = theta12.value*self.deg
                
                if self.mh==MassHierarchy.NORMAL:
                    dNdE = np.cos(theta12)**2*dNdE_nuebar+np.sin(theta12)**2*dNdE_nux
                if self.mh==MassHierarchy.INVERTED:
                    dNdE = np.sin(theta12)**2*dNdE_nuebar+np.cos(theta12)**2*dNdE_nux
            else:
                if self.SNmodel=='Bollig_2016':
                    model = ccsn.Bollig_2016(progenitor_mass=float(self.config['SUPERNOVA']['mass'])*u.solMass)
                elif self.SNmodel=='Fornax_2021':
                    model = ccsn.Fornax_2021(progenitor_mass=float(self.config['SUPERNOVA']['mass'])*u.solMass)
                elif self.SNmodel=='Kuroda_2020':
                    model = ccsn.Kuroda_2020(rotational_velocity=float(self.config['SUPERNOVA']['rot_vel'])*u.rad/u.second,magnetic_field_exponent=float(self.config['SUPERNOVA']['mag_exp']))
                elif self.SNmodel=='Nakazato_2013':
                    model = ccsn.Nakazato_2013(progenitor_mass=float(self.config['SUPERNOVA']['mass'])*u.solMass,revival_time=float(self.config['SUPERNOVA']['rev_time'])*u.ms,metallicity=float(self.config['SUPERNOVA']['metallicity']),eos=self.config['SUPERNOVA']['eos'])
                elif self.SNmodel=='Tamborra_2014':
                    model = ccsn.Tamborra_2014(progenitor_mass=float(self.config['SUPERNOVA']['mass'])*u.solMass)
                    
                xform = AdiabaticMSW(mh=self.mh)

                t_list = model.time.value*self.s
                N_E = len(E)
                N_t = len(t_list)
                dNdtdE = np.empty((N_t,N_E))
                for i in np.arange(N_t):
                    t = t_list[i]
                    dNdtdE[i] = model.get_transformed_spectra((t/self.s)*u.s,
                                                              (E/self.MeV)*u.MeV,
                                                              xform)[self.flavor].value \
                        *self.erg**-1*self.s**-1
                dNdE = simps(dNdtdE,x=t_list,axis=0)
                
            return dNdE
    


    
    def dNdE_calc_IMFweighed(self, E):
        """ 
        Note for Fornax_2021, if late phase is included we will not add in 12 and 15M progenitors since they do not expode
        """
        
        xform = AdiabaticMSW(mh=self.mh)
        N_E = len(E)
        if self.config['LATEPHASE']['late_dNdE']=='T':
            if self.SNmodel=='Fornax_2021':
                # Errors with 20 and 26.99 for some reason in SNEWPY.
                model_list = [13, 14, 16, 17, 18, 19, 21, 22, 23, 25, 26]
                F21initmasses = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 26.99]
                F21revtimes = [0, 409.4, 429.9, 0, 255.6, 347.0, 272.8, 309.5, 332.6, 344.2, 365.1, 389.8, 334.6, 315.2, 338.9]
                F21finalmasses = [0, 1.88, 1.80, 0, 1.74, 2.04, 1.79, 1.86, 2.09, 2.25, 2.05, 2.03, 2.10, 2.14, 2.11]
                
                dNdE = np.empty((N_E,len(model_list)))
                for modelname in model_list:
                    revtime = F21revtimes[F21initmasses.index(modelname)]
                    finalmass = F21finalmasses[F21initmasses.index(modelname)]
                    
                    model_id = model_list.index(modelname)
                    model = ccsn.Fornax_2021(progenitor_mass=modelname*u.solMass)
                    t_list = model.time.value[:self.find_index(list(model.time.value),revtime/1000)]*self.s
                    N_t = len(t_list)
                    dNdtdE = np.empty((N_t,N_E))
                    for i in np.arange(N_t):
                        t = t_list[i]
                        dNdtdE[i] = model.get_transformed_spectra((t/self.s)*u.s,
                                                                    (E/self.MeV)*u.MeV,
                                                                    xform)[self.flavor].value \
                            *self.erg**-1*self.s**-1
                            
                    Etot = (self.lateliberatedenergy(finalmass,revtime,Flavor.NU_E)*self.erg)+\
                        (self.lateliberatedenergy(finalmass,revtime,Flavor.NU_E_BAR)*self.erg)+\
                        4*(self.lateliberatedenergy(finalmass,revtime,Flavor.NU_X)*self.erg)
                    Tnuebar = self.energy_to_temp(self.latemeanenergy(finalmass,revtime,Flavor.NU_E_BAR),2.3)*self.MeV
                    Tnux = self.energy_to_temp(self.latemeanenergy(finalmass,revtime,Flavor.NU_X),2.3)*self.MeV
            
                    dNdE2_nuebar = Etot/6.*120./(7.*np.pi**4)*E**2/Tnuebar**4 \
                        /(np.exp(E/Tnuebar)+1.)
                    dNdE2_nux = Etot/6.*120./(7.*np.pi**4)*E**2/Tnux**4 \
                        /(np.exp(E/Tnux)+1.)
                    
                    theta12,theta13,theta23 = MixingParameters(mh=self.mh).get_mixing_angles()
                    theta12 = theta12.value*self.deg
                    
                    if self.mh==MassHierarchy.NORMAL:
                        dNdE2 = np.cos(theta12)**2*dNdE2_nuebar+np.sin(theta12)**2*dNdE2_nux
                    if self.mh==MassHierarchy.INVERTED:
                        dNdE2 = np.sin(theta12)**2*dNdE2_nuebar+np.cos(theta12)**2*dNdE2_nux
                    
                    dNdE[:,model_id] = simps(dNdtdE,x=t_list,axis=0) + dNdE2
                    
                Ml = np.append(8,np.array([(model_list[i]+model_list[i+1])/2 for i in range(len(model_list)-1)]))*self.Msun
                Mu = np.append(np.array([(model_list[i]+model_list[i+1])/2 for i in range(len(model_list)-1)]),125)*self.Msun
                    
            elif self.SNmodel=='Nakazato_2013':
                model_list = [13, 20, 30, 50]
                
                N13inits = [{13, 0.02, 100}, {13, 0.02, 200}, {13, 0.02, 300}, {20, 0.02, 100}, {20, 0.02, 200}, {20, 0.02, 300}, {30, 0.02, 100}, {30, 0.02, 200}, {30, 0.02, 300}, {50, 0.02, 100}, {50, 0.02, 200}, {50, 0.02, 300}, {13, 0.004, 100}, {13, 0.004, 200}, {13, 0.004, 300}, {20, 0.004, 100}, {20, 0.004, 200}, {20, 0.004, 300}, {50, 0.004, 100}, {50, 0.004, 200}, {50, 0.004, 300}]
                N13finalmasses = [1.50, 1.59, 1.64, 1.47, 1.54, 1.57, 1.62, 1.83, 1.98, 1.67, 1.79, 1.87, 1.50, 1.58, 1.63, 1.63, 1.73, 1.77, 1.67, 1.79, 1.91]
                
                dNdE = np.empty((N_E,len(model_list)))
                for modelname in model_list:
                    revtime = float(self.config['SUPERNOVA']['rev_time'])
                    finalmass = N13finalmasses[N13inits.index({modelname,float(self.config['SUPERNOVA']['metallicity']),float(self.config['SUPERNOVA']['rev_time'])})]
                    
                    model_id = model_list.index(modelname)
                    model = ccsn.Nakazato_2013(progenitor_mass=modelname*u.solMass,revival_time=float(self.config['SUPERNOVA']['rev_time'])*u.ms,metallicity=float(self.config['SUPERNOVA']['metallicity']),eos=self.config['SUPERNOVA']['eos'])
                    t_list = model.time.value[:self.find_index(list(model.time.value),revtime/1000)]*self.s
                    N_t = len(t_list)
                    dNdtdE = np.empty((N_t,N_E))
                    for i in np.arange(N_t):
                        t = t_list[i]
                        dNdtdE[i] = model.get_transformed_spectra((t/self.s)*u.s,
                                                                    (E/self.MeV)*u.MeV,
                                                                    xform)[self.flavor].value \
                            *self.erg**-1*self.s**-1
                            
                    Etot = (self.lateliberatedenergy(finalmass,revtime,Flavor.NU_E)*self.erg)+\
                        (self.lateliberatedenergy(finalmass,revtime,Flavor.NU_E_BAR)*self.erg)+\
                        4*(self.lateliberatedenergy(finalmass,revtime,Flavor.NU_X)*self.erg)
                    Tnuebar = self.energy_to_temp(self.latemeanenergy(finalmass,revtime,Flavor.NU_E_BAR),2.3)*self.MeV
                    Tnux = self.energy_to_temp(self.latemeanenergy(finalmass,revtime,Flavor.NU_X),2.3)*self.MeV
            
                    dNdE2_nuebar = Etot/6.*120./(7.*np.pi**4)*E**2/Tnuebar**4 \
                        /(np.exp(E/Tnuebar)+1.)
                    dNdE2_nux = Etot/6.*120./(7.*np.pi**4)*E**2/Tnux**4 \
                        /(np.exp(E/Tnux)+1.)
                    
                    theta12,theta13,theta23 = MixingParameters(mh=self.mh).get_mixing_angles()
                    theta12 = theta12.value*self.deg
                    
                    if self.mh==MassHierarchy.NORMAL:
                        dNdE2 = np.cos(theta12)**2*dNdE2_nuebar+np.sin(theta12)**2*dNdE2_nux
                    if self.mh==MassHierarchy.INVERTED:
                        dNdE2 = np.sin(theta12)**2*dNdE2_nuebar+np.cos(theta12)**2*dNdE2_nux
                    
                    dNdE[:,model_id] = simps(dNdtdE,x=t_list,axis=0) + dNdE2
                    
                Ml = np.append(8,np.array([(model_list[i]+model_list[i+1])/2 for i in range(len(model_list)-1)]))*self.Msun
                Mu = np.append(np.array([(model_list[i]+model_list[i+1])/2 for i in range(len(model_list)-1)]),125)*self.Msun
            else:
                raise Exception('Only implemented for Fornax 2021 and Nakazato 2013 models')
        else:
             if self.SNmodel=='Bollig_2016':
                 model_list = [11.2, 27]
                 dNdE = np.empty((N_E,2))
                 for modelname in model_list:
                     model_id = model_list.index(modelname)
                     model = ccsn.Bollig_2016(progenitor_mass=modelname*u.solMass)
                     t_list = model.time.value*self.s
                     N_t = len(t_list)
                     dNdtdE = np.empty((N_t,N_E))
                     for i in np.arange(N_t):
                         t = t_list[i]
                         dNdtdE[i] = model.get_transformed_spectra((t/self.s)*u.s,
                                                                     (E/self.MeV)*u.MeV,
                                                                     xform)[self.flavor].value \
                             *self.erg**-1*self.s**-1
                     dNdE[:,model_id] = simps(dNdtdE,x=t_list,axis=0)
                 
                 # Choosing mass bins in between initial progenitor masses
                 Ml = np.append(8,np.array([(model_list[i]+model_list[i+1])/2 for i in range(len(model_list)-1)]))*self.Msun
                 Mu = np.append(np.array([(model_list[i]+model_list[i+1])/2 for i in range(len(model_list)-1)]),125)*self.Msun
             elif self.SNmodel=='Fornax_2021':
                 # Now includes 12 and 15, which don't explode.
                 model_list = [12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26]
                 dNdE = np.empty((N_E,len(model_list)))
                 for modelname in model_list:
                     model_id = model_list.index(modelname)
                     model = ccsn.Fornax_2021(progenitor_mass=modelname*u.solMass)
                     t_list = model.time.value*self.s
                     N_t = len(t_list)
                     dNdtdE = np.empty((N_t,N_E))
                     for i in np.arange(N_t):
                         t = t_list[i]
                         dNdtdE[i] = model.get_transformed_spectra((t/self.s)*u.s,
                                                                     (E/self.MeV)*u.MeV,
                                                                     xform)[self.flavor].value \
                             *self.erg**-1*self.s**-1
                     dNdE[:,model_id] = simps(dNdtdE,x=t_list,axis=0)
                     
                 Ml = np.append(8,np.array([(model_list[i]+model_list[i+1])/2 for i in range(len(model_list)-1)]))*self.Msun
                 Mu = np.append(np.array([(model_list[i]+model_list[i+1])/2 for i in range(len(model_list)-1)]),125)*self.Msun
             elif self.SNmodel=='Kuroda_2020':
                 raise Exception('There is only one progenitor mass model for Kuroda 2020')
             elif self.SNmodel=='Nakazato_2013':
                 model_list = [13, 20, 30, 50]
                 dNdE = np.empty((N_E,len(model_list)))
                 for modelname in model_list:
                     model_id = model_list.index(modelname)
                     model = ccsn.Nakazato_2013(progenitor_mass=modelname*u.solMass,revival_time=float(self.config['SUPERNOVA']['rev_time'])*u.ms,metallicity=float(self.config['SUPERNOVA']['metallicity']),eos=self.config['SUPERNOVA']['eos'])
                     t_list = model.time.value*self.s
                     N_t = len(t_list)
                     dNdtdE = np.empty((N_t,N_E))
                     for i in np.arange(N_t):
                         t = t_list[i]
                         dNdtdE[i] = model.get_transformed_spectra((t/self.s)*u.s,
                                                                     (E/self.MeV)*u.MeV,
                                                                     xform)[self.flavor].value \
                             *self.erg**-1*self.s**-1
                     dNdE[:,model_id] = simps(dNdtdE,x=t_list,axis=0)
                     
                 Ml = np.append(8,np.array([(model_list[i]+model_list[i+1])/2 for i in range(len(model_list)-1)]))*self.Msun
                 Mu = np.append(np.array([(model_list[i]+model_list[i+1])/2 for i in range(len(model_list)-1)]),125)*self.Msun
             elif self.SNmodel=='Tamborra_2014':
                 model_list = [20, 27]
                 dNdE = np.empty((N_E,2))
                 for modelname in model_list:
                     model_id = model_list.index(modelname)
                     model = ccsn.Tamborra_2014(progenitor_mass=modelname*u.solMass)
                     t_list = model.time.value*self.s
                     N_t = len(t_list)
                     dNdtdE = np.empty((N_t,N_E))
                     for i in np.arange(N_t):
                         t = t_list[i]
                         dNdtdE[i] = model.get_transformed_spectra((t/self.s)*u.s,
                                                                     (E/self.MeV)*u.MeV,
                                                                     xform)[self.flavor].value \
                             *self.erg**-1*self.s**-1
                     dNdE[:,model_id] = simps(dNdtdE,x=t_list,axis=0)
                 
                 Ml = np.append(8,np.array([(model_list[i]+model_list[i+1])/2 for i in range(len(model_list)-1)]))*self.Msun
                 Mu = np.append(np.array([(model_list[i]+model_list[i+1])/2 for i in range(len(model_list)-1)]),125)*self.Msun   

        
        M  = np.logspace(np.log10(Ml/self.Msun),np.log10(Mu/self.Msun),1001)*self.Msun
        Mtot = np.logspace(np.log10(8.),np.log10(125.),1001)*self.Msun
        M = np.expand_dims(M,axis=1)
        denominator = simps(self.IMF(Mtot)*Mtot,x=np.log(Mtot))
        numerator = np.sum(simps(self.IMF(M)*M*dNdE,x=np.log(M),axis=0),axis=-1)
        dNdE_weighed = numerator/denominator
        
        return dNdE_weighed
        
        
        
    def dNdE_calc_BHweighed(self, E):
        """ 
        Added if wanted to weigh spectrum with failed CCSNe (i.e. BH forming cases) 
        """        
        if self.SNmodel=='FermiDirac':
            Etot = float(self.config['SUPERNOVA']['Etot_erg'])*self.erg
            if self.flavor==Flavor.NU_E_BAR:
                Tnuebar = float(self.config['SUPERNOVA']['Tnuebar_MeV'])*self.MeV
                Tnux = float(self.config['SUPERNOVA']['Tnux_MeV'])*self.MeV

            dNdE_nuebar = Etot/6.*120./(7.*np.pi**4)*E**2/Tnuebar**4 \
                /(np.exp(E/Tnuebar)+1.)
            dNdE_nux = Etot/6.*120./(7.*np.pi**4)*E**2/Tnux**4 \
                /(np.exp(E/Tnux)+1.)
            
            theta12,theta13,theta23 = MixingParameters(mh=self.mh).get_mixing_angles()
            theta12 = theta12.value*self.deg
            
            if self.mh==MassHierarchy.NORMAL:
                dNdE = np.cos(theta12)**2*dNdE_nuebar+np.sin(theta12)**2*dNdE_nux
            if self.mh==MassHierarchy.INVERTED:
                dNdE = np.sin(theta12)**2*dNdE_nuebar+np.cos(theta12)**2*dNdE_nux
        elif self.SNmodel=='PinchedFermiDirac':
            Etot = float(self.config['SUPERNOVA']['Etot_erg'])*self.erg
            Enuebar = float(self.config['SUPERNOVA']['Enuebar_MeV'])*self.MeV
            Enux = float(self.config['SUPERNOVA']['Enux_MeV'])*self.MeV
            Alpha = float(self.config['SUPERNOVA']['alpha'])
            
            dNdE_nuebar = self.pinched_FD(Enuebar, Etot/6, Alpha, E)
            dNdE_nux = self.pinched_FD(Enux, Etot/6, Alpha, E)
            
            theta12,theta13,theta23 = MixingParameters(mh=self.mh).get_mixing_angles()
            theta12 = theta12.value*self.deg
            
            if self.mh==MassHierarchy.NORMAL:
                dNdE = np.cos(theta12)**2*dNdE_nuebar+np.sin(theta12)**2*dNdE_nux
            if self.mh==MassHierarchy.INVERTED:
                dNdE = np.sin(theta12)**2*dNdE_nuebar+np.cos(theta12)**2*dNdE_nux    
        else:
            if self.SNmodel=='Bollig_2016':
                model = ccsn.Bollig_2016(progenitor_mass=float(self.config['SUPERNOVA']['mass'])*u.solMass)
            elif self.SNmodel=='Fornax_2021':
                model = ccsn.Fornax_2021(progenitor_mass=float(self.config['SUPERNOVA']['mass'])*u.solMass)
            elif self.SNmodel=='Kuroda_2020':
                model = ccsn.Kuroda_2020(rotational_velocity=float(self.config['SUPERNOVA']['rot_vel'])*u.rad/u.second,magnetic_field_exponent=float(self.config['SUPERNOVA']['mag_exp']))
            elif self.SNmodel=='Nakazato_2013':
                model = ccsn.Nakazato_2013(progenitor_mass=float(self.config['SUPERNOVA']['mass'])*u.solMass,revival_time=float(self.config['SUPERNOVA']['rev_time'])*u.ms,metallicity=float(self.config['SUPERNOVA']['metallicity']),eos=self.config['SUPERNOVA']['eos'])
            elif self.SNmodel=='Tamborra_2014':
                model = ccsn.Tamborra_2014(progenitor_mass=float(self.config['SUPERNOVA']['mass'])*u.solMass)
                
            xform = AdiabaticMSW(mh=self.mh)

            t_list = model.time.value*self.s
            N_E = len(E)
            N_t = len(t_list)
            dNdtdE = np.empty((N_t,N_E))
            for i in np.arange(N_t):
                t = t_list[i]
                dNdtdE[i] = model.get_transformed_spectra((t/self.s)*u.s,
                                                          (E/self.MeV)*u.MeV,
                                                          xform)[self.flavor].value \
                    *self.erg**-1*self.s**-1
            dNdE = simps(dNdtdE,x=t_list,axis=0)
            
        model2 = ccsn.Nakazato_2013(progenitor_mass=30*u.solMass,revival_time=0*u.ms,metallicity=0.004,eos=self.config['FAILEDSNE']['failed_eos'])
        t_list2 = model2.time.value*self.s
        
        xform = AdiabaticMSW(mh=self.mh)
        N_E = len(E)
        N_t2 = len(t_list2)
        dNdtdE2 = np.empty((N_t2,N_E))
        for i in np.arange(N_t2):
            t2 = t_list2[i]
            dNdtdE2[i] = model2.get_transformed_spectra((t2/self.s)*u.s,
                                                      (E/self.MeV)*u.MeV,
                                                      xform)[self.flavor].value \
                *self.erg**-1*self.s**-1
        dNdE2 = simps(dNdtdE2,x=t_list2,axis=0)
            
        return (1-float(self.config['FAILEDSNE']['BH_fraction']))*dNdE + (float(self.config['FAILEDSNE']['BH_fraction']))*dNdE2




class late_phase_energetics(units_and_constants):
    """ 
    Calculate late phase mean and liberated energies using one of four estimation methods from Ekanger et al. 2022 arXiv:2206.05299, with inputs from the initialization file 
    """
    
    def __init__(self, inifile):
        units_and_constants.__init__(self)
        self.flavor = Flavor[self.config['NEUTRINO']['flavor']]
        self.method = self.config['LATEPHASE']['method']
    
    def suwaanalytict0(self,g,beta,fm,rad_km,Etot_erg):
        """
        Define t0 needed for Suwa et al. 2021 arXiv: 2008.07070, analytical function (these are independent of neutrino flavor)
        """
        return (g*beta/3)**(4/5)*210*(fm/1.4)**(6/5)*(rad_km/10)**(-6/5)*(Etot_erg/1e52)**(-1/5)
    
    def suwameanenergy(self,t,g,beta,fm,rad_km,Etot_erg):
        """
        Define mean energy for analytical function
        """
        return (g*beta/3)*16*(fm/1.4)**(3/2)*(rad_km/10)**(-2)*((t+self.suwaanalytict0(g,beta,fm,rad_km,Etot_erg))/100)**(-3/2)
    
    def suwaluminosity(self,t,g,beta,fm,rad_km,Etot_erg):
        """
        Define luminosity for analytical function
        """
        return (g*beta/3)**4*(3.3e51)*(fm/1.4)**6*(rad_km/10)**(-6)*((t+self.suwaanalytict0(g,beta,fm,rad_km,Etot_erg))/100)**(-6)
    
    def integmean(self,fm,rt,g=None,beta=None,rad_km=None,Etot_erg=None):
        """
        Integrate analytic mean energy (get number-weighed average energy)
        """
        result = 0
        # Integrate with default values
        if g==None and beta==None and rad_km==None and Etot_erg==None:
            int1=integrate.quad(lambda x: self.suwameanenergy(x,0.07,26.5,fm,12,3.2e53)*self.suwaluminosity(x,0.07,26.5,fm,12,3.2e53)/self.suwameanenergy(x,0.07,26.5,fm,12,3.2e53),rt/1000,1e5)[0]
            int2=integrate.quad(lambda x: self.suwaluminosity(x,0.07,26.5,fm,12,3.2e53)/self.suwameanenergy(x,0.07,26.5,fm,12,3.2e53),rt/1000,1e5)[0]
            result = int1/int2
        # Integrate with input values
        else:
            int1=integrate.quad(lambda x: self.suwameanenergy(x,g,beta,fm,rad_km,Etot_erg)*self.suwaluminosity(x,g,beta,fm,rad_km,Etot_erg)/self.suwameanenergy(x,g,beta,fm,rad_km,Etot_erg),rt/1000,1e5)[0]
            int2=integrate.quad(lambda x: self.suwaluminosity(x,g,beta,fm,rad_km,Etot_erg)/self.suwameanenergy(x,g,beta,fm,rad_km,Etot_erg),rt/1000,1e5)[0]
            result = int1/int2
        
        return result

    def integlum(self,fm,rt,g=None,beta=None,rad_km=None,Etot_erg=None):
        """
        Integrate analytic luminosity (get energy liberated)
        """
        # Integrate with default values
        if g==None and beta==None and rad_km==None and Etot_erg==None:
            return integrate.quad(lambda x: self.suwaluminosity(x,0.07,26.5,fm,12,3.2e53),rt/1000,1e5)[0]
        # Integrate with input values
        else:
            return integrate.quad(lambda x: self.suwaluminosity(x,g,beta,fm,rad_km,Etot_erg),rt/1000,1e5)[0]
        
    def latemeanenergy(self, fm, rt, flvr):
        """
        Returns the mean energy (in MeV) calculated from revival time to ~20 seconds (fm in solar masses, rt in ms)
        """
        result = 0
        if flvr == Flavor.NU_E:
            if self.method == 'Corr':
                result = 0.999*fm + (-1.17e-3)*rt + 6.99
            elif self.method == 'RenormShen':
                result = 1.13*(0.999*fm + (-1.17e-3)*rt + 6.99)
            elif self.method == 'RenormLS':
                result = 1.28*(0.999*fm + (-1.17e-3)*rt + 6.99)
            elif self.method == 'Analyt':
                # Evaluate if default values will be used
                if self.config['LATEPHASE']['default'] == 'T':
                    result = self.integmean(fm,rt)
                # If default is false, use input values from .ini file
                else:
                    result = self.integmean(fm,rt,float(self.config['LATEPHASE']['g']),float(self.config['LATEPHASE']['beta']),float(self.config['LATEPHASE']['rad_km']),float(self.config['LATEPHASE']['Etot_erg']))
        if flvr == Flavor.NU_E_BAR:
            if self.method == 'Corr':
                result = 0.887*fm + (-2.25e-3)*rt + 9.06
            elif self.method == 'RenormShen':
                result = 1.15*(0.887*fm + (-2.25e-3)*rt + 9.06)
            elif self.method == 'RenormLS':
                result = 1.32*(0.887*fm + (-2.25e-3)*rt + 9.06)
            elif self.method == 'Analyt':
                if self.config['LATEPHASE']['default'] == 'T':
                    result = self.integmean(fm,rt)
                else:
                    result = self.integmean(fm,rt,float(self.config['LATEPHASE']['g']),float(self.config['LATEPHASE']['beta']),float(self.config['LATEPHASE']['rad_km']),float(self.config['LATEPHASE']['Etot_erg']))
        else:
            if self.method == 'Corr':
                result = 0.892*fm + (-2.36e-3)*rt + 10.1
            elif self.method == 'RenormShen':
                result = 1.00*(0.892*fm + (-2.36e-3)*rt + 10.1)
            elif self.method == 'RenormLS':
                result = 1.16*(0.892*fm + (-2.36e-3)*rt + 10.1)
            elif self.method == 'Analyt':
                if self.config['LATEPHASE']['default'] == 'T':
                    result = self.integmean(fm,rt)
                else:
                    result = self.integmean(fm,rt,float(self.config['LATEPHASE']['g']),float(self.config['LATEPHASE']['beta']),float(self.config['LATEPHASE']['rad_km']),float(self.config['LATEPHASE']['Etot_erg']))
        
        return result
       
    def lateliberatedenergy(self, fm, rt, flvr):
        """
        Returns the liberated energy (in erg) calculated from revival time to ~20 seconds (fm in solar masses, rt in ms)
        """
        result = 0
        if flvr == Flavor.NU_E:
            if self.method == 'Corr':
                result = 10**(0.317*fm + (-5.54e-4)*rt + 51.92)
            elif self.method == 'RenormShen':
                result = 2.57*(10**(0.317*fm + (-5.54e-4)*rt + 51.92))
            elif self.method == 'RenormLS':
                result = 2.33*(10**(0.317*fm + (-5.54e-4)*rt + 51.92))
            elif self.method == 'Analyt':
                # Evaluate if default values will be used
                if self.config['LATEPHASE']['default'] == 'T':
                    result = self.integlum(fm,rt)
                # If default is false, use input values from .ini file
                else:
                    result = self.integlum(fm,rt,float(self.config['LATEPHASE']['g']),float(self.config['LATEPHASE']['beta']),float(self.config['LATEPHASE']['rad_km']),float(self.config['LATEPHASE']['Etot_erg']))
        if flvr == Flavor.NU_E_BAR:
            if self.method == 'Corr':
                result = 10**(0.375*fm + (-6.04e-4)*rt + 51.85)
            elif self.method == 'RenormShen':
                result = 2.71*(10**(0.375*fm + (-6.04e-4)*rt + 51.85))
            elif self.method == 'RenormLS':
                result = 2.46*(10**(0.375*fm + (-6.04e-4)*rt + 51.85))
            elif self.method == 'Analyt':
                if self.config['LATEPHASE']['default'] == 'T':
                    result = self.integlum(fm,rt)
                else:
                    result = self.integlum(fm,rt,float(self.config['LATEPHASE']['g']),float(self.config['LATEPHASE']['beta']),float(self.config['LATEPHASE']['rad_km']),float(self.config['LATEPHASE']['Etot_erg']))
        else:
            if self.method == 'Corr':
                result = 10**(0.412*fm + (-3.97e-4)*rt + 51.92)
            elif self.method == 'RenormShen':
                result = 1.72*(10**(0.412*fm + (-3.97e-4)*rt + 51.87))
            elif self.method == 'RenormLS':
                result = 1.63*(10**(0.412*fm + (-3.97e-4)*rt + 51.87))
            elif self.method == 'Analyt':
                if self.config['LATEPHASE']['default'] == 'T':
                    result = self.integlum(fm,rt)
                else:
                    result = self.integlum(fm,rt,float(self.config['LATEPHASE']['g']),float(self.config['LATEPHASE']['beta']),float(self.config['LATEPHASE']['rad_km']),float(self.config['LATEPHASE']['Etot_erg']))
        
        return result

   
    

class DSNB(cosmology, SN_neutrino_spectrum, late_phase_energetics):
    """
    Calculate DSNB flux spectrum
    """
    
    
    def __init__(self, inifile):
        cosmology.__init__(self)
        SN_neutrino_spectrum.__init__(self,inifile)
        late_phase_energetics.__init__(self,inifile)
    
    
    def DSNB_dFdE(self, E):
        z = np.linspace(0.,5.,1000)
        z_2D = z.reshape(-1,1)
        integrand = 1./self.Hubble(z_2D)*self.R_SN(z_2D) \
            *self.dNdE_func((1.+z_2D)*E)
        dFdE = simps(integrand,x=z,axis=0)
        return dFdE
    