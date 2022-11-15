import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import interp1d
from snewpy.models import ccsn
from snewpy.neutrino import MassHierarchy, Flavor, MixingParameters
from snewpy.flavor_transformation import AdiabaticMSW
from astropy import units as u
import configparser



class units_and_constants:
    
    
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
    
    
    def __init__(self):
        units_and_constants.__init__(self)
        
        
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
        return R_SF

    
    def IMF(self, M, model='Salpeter'):
        """
        model: 'Salpeter' (default), 'Kroupa', or 'BG'
        """
        if model=='Salpeter':
            xi1 = 2.35
            xi2 = 2.35
        elif model=='Kroupa':
            xi1 = 1.3
            xi2 = 2.3
        elif model=='BG':
            xi1 = 1.5
            xi2 = 2.15
        
        phi = np.where(M>0.5*self.Msun,(M/(0.5*self.Msun))**-xi2,
                       (M/(0.5*self.Msun))**-xi1)
        return phi
    
    
    def R_SN(self, z, IMFmodel='Salpeter'):
        M1 = np.logspace(-1,2,1000)*self.Msun
        M2 = np.logspace(np.log10(8.),2,1000)*self.Msun
        denominator = simps(M1**2*self.IMF(M1,IMFmodel),x=np.log(M1))
        numerator   = simps(M2*self.IMF(M2,IMFmodel),x=np.log(M2))
        conversion  = numerator/denominator
        R_SN = conversion*self.R_SF(z)
        return R_SN
    
        

        
class SN_neutrino_spectrum(units_and_constants):
    
    
    def __init__(self, inifile):
        units_and_constants.__init__(self)
        self.config = configparser.ConfigParser()
        self.config.read(inifile)
        if self.config['NEUTRINO']['flavor']=='NU_E_BAR':
            self.flavor = Flavor.NU_E_BAR
        if self.config['NEUTRINO']['mh']=='Normal':
            self.mh = MassHierarchy.NORMAL
        elif self.config['NEUTRINO']['mh']=='Inverted':
            self.mh = MassHierarchy.INVERTED
        self.SNmodel = self.config['SUPERNOVA']['authors']
        E = np.linspace(0.,100.,2001)*self.MeV
        self.dNdE_func = interp1d(E,self.dNdE_calc(E),
                                  bounds_error=False,fill_value=0.)
        
        
    def dNdE_calc(self, E):
        
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
            
        else:
            self.modelname = self.config['SUPERNOVA']['model']

            if self.SNmodel=='Bollig_2016':
                self.modelname2 = self.config['SUPERNOVA']['model2']
                model = ccsn.Bollig_2016('SNEWPY_models/'+self.SNmodel+'/'+self.modelname,
                                         self.modelname2)
            elif self.SNmodel=='Fornax_2021':
                model = ccsn.Fornax_2021('SNEWPY_models/'+self.SNmodel+'/'+self.modelname)
            elif self.SNmodel=='Kuroda_2020':
                model = ccsn.Kuroda_2020('SNEWPY_models/'+self.SNmodel+'/'+self.modelname)
            elif self.SNmodel=='Nakazato_2013':
                model = ccsn.Nakazato_2013('SNEWPY_models/'+self.SNmodel+'/'+self.modelname)
            elif self.SNmodel=='Tamborra_2014':
                model = ccsn.Tamborra_2014('SNEWPY_models/'+self.SNmodel+'/'+self.modelname)
                
            xform = AdiabaticMSW(mh=self.mh)

            t_list = model.time.value*self.s
            #print(t_list/self.s)
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
    
    
    
    
class DSNB(cosmology, supernova_rate, SN_neutrino_spectrum):
    
    
    def __init__(self, inifile):
        cosmology.__init__(self)
        supernova_rate.__init__(self)
        SN_neutrino_spectrum.__init__(self,inifile)
    
    
    def DSNB_dFdE(self, E, IMFmodel='Salpeter'):
        z = np.linspace(0.,5.,1000)
        z_2D = z.reshape(-1,1)
        integrand = 1./self.Hubble(z_2D)*self.R_SN(z_2D,IMFmodel) \
            *self.dNdE_func((1.+z_2D)*E)
        dFdE = simps(integrand,x=z,axis=0)
        return dFdE
    