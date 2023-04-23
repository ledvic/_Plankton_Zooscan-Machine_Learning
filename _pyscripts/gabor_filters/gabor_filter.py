from scipy.special import erfcinv
import numpy as np
# __all__ = ["my_custom_utils"]

class GaborFilter:
    # constructor
    def __init__(self):
        # private members
        self.__Frequency = None # frequency of the filter
        self.__MaxFrequency = None # maximum frequency of the whole filter bank
        self.__Orientation = None # orientation of the filter
        self.__Envelope = None # the Gaussian part of the Gabor filter equations
        self.__FrequencyFilter = None # the Gabor filter on frequency domain
        self.__SpatialFilter = None # the Gabor filter on spatial domain
        self.__Domain = 1 #  "1" is intepreted as frequency domain; "0" encoded spatial domain

    @property # first decorate the getter method
    def frequency(self): # This getter method name is *the* name
        return self.__Frequency
    @frequency.setter    # the property decorates with `.setter` now
    def frequency(self, value):   # name, e.g. "attribute", is the same
        self.__Frequency = value   # the "value" name isn't special
    
    @property 
    def max_frequency(self): 
        return self.__MaxFrequency
    @max_frequency.setter 
    def max_frequency(self, value):   
        self.__MaxFrequency = value
        
    @property 
    def orientation(self): 
        return self.__Orientation
    @orientation.setter 
    def orientation(self, value):   
        self.__Orientation = value   
        
    @property 
    def envelope(self): 
        return self.__Envelope
    @envelope.setter 
    def envelope(self, value):   
        self.__Envelope = value   

    @property 
    def FrequencyDomainGaborFilter(self): 
        return self.__FrequencyFilter
    @FrequencyDomainGaborFilter.setter 
    def FrequencyDomainGaborFilter(self, value):   
        self.__FrequencyFilter = value   

    @property 
    def SpatialDomainGaborFilter(self): 
        return self.__SpatialFilter
    @SpatialDomainGaborFilter.setter 
    def SpatialDomainGaborFilter(self, value):   
        self.__SpatialFilter = value       
    
    @property 
    def FilterDomain(self): 
        return self.__Domain
    @FilterDomain.setter 
    def FilterDomain(self, value):   
        self.__Domain = value
        
    @staticmethod    
    def __norminv(self, p, mu, sigma):
    
        x0 = np.multiply( -np.sqrt(2), (erfcinv(2 * p)) );
        x = np.multiply(sigma, x0) + mu;

        return x
    
    
    def __compute_fhigh(self, a, b):
        
        f0 = self.frequency
        
        d = f0;

        if b > a:
            
            foo = -( (a**2) * d )/(a**2 - b**2)
            
            if foo < a:
                fhigh2 = np.sqrt( (d + foo)**2 + (b/a * np.sqrt(a**2 - foo**2))**2 )
            else:
                fhigh2 = d + a;
        
        else:
            fhigh2 = d + a; 

        if fhigh2 > 0.5:
            fhigh2 = 0.5; 

        self.max_frequency = fhigh2
    
    @staticmethod
    def __ellipsoid_envelope_point(self, a, b, c):

        x = (c * (a**2)) / np.sqrt(b**2 + c**2 * a**2)
        y = b / a * np.sqrt(a**2 - x**2);

        return np.array([x,y])
    
    
    def __accurate_envelope_f(self, a, b):
        
        f0 = self.frequency
        theta = self.orientation
        
        if np.mod(theta, np.pi/2)!=0:

            # solve points with slopes -tan(pi/2-theta) and tan(theta)
            x1y1 = self.__ellipsoid_envelope_point(a, b, -np.tan(np.pi/2-theta))
            x2y2 = self.__ellipsoid_envelope_point(a, b, np.tan(theta))

            envelope = np.array([x1y1, -x1y1, x2y2, -x2y2])

            # shift by f0
            envelope = envelope + np.tile([f0, 0],[4,1])

            # rotate by theta
            envelope = np.matmul(envelope, np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]]))

            xmin = min(envelope[:,0]);
            xmax = max(envelope[:,0]);
            ymin = min(envelope[:,1]);
            ymax = max(envelope[:,1]);

            envelope = np.real([xmin, xmax, ymin, ymax])

        else: 

            envelope = np.array([[f0 - a, 0], [f0 + a, 0] , [f0, b], [f0, -b]])

            envelope = np.matmul(envelope, np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]]))

            xmin = min(envelope[:,0]);
            xmax = max(envelope[:,0]);
            ymin = min(envelope[:,1]);
            ymax = max(envelope[:,1]);

            envelope=np.real([xmin, xmax, ymin, ymax])

        return envelope
    
    
    def transform_from_frequency_to_spatial_domain(self, row, col):
        
        envelope = self.envelope
        FrequencyGaborFilter = self.FrequencyDomainGaborFilter
        
        M1 = col
        M2 = row

        lx = envelope[1] - envelope[0];
        ly = envelope[3] - envelope[2];

        xx = np.int32(np.mod( np.arange(0,lx + 1) + envelope[0] + M1 , M1 ) + 1)-1;
        yy = np.int32(np.mod( np.arange(0,ly + 1) + envelope[2] + M2 , M2 ) + 1)-1;

        f2_ = np.zeros((M2,M1), dtype=np.float64)

        for y in range(0,len(yy)):
            for x in range(0,len(xx)):
                xf2 = xx[x] 
                yf2 = yy[y] 

                f2_[yf2,xf2] = FrequencyGaborFilter[y,x]

        f2 = np.fft.ifftshift( np.fft.ifft2(f2_) )

        self.SpatialDomainGaborFilter = f2
    
    
    def compute_a_gabor_filter(self, gamma, eta, row, col, domain=1):
        
        Nx = col
        Ny = row
        
        f0 = self.frequency
        theta = self.orientation
        
        pf = np.sqrt(0.99) #energy to include in the filters
        alpha = f0 / gamma
        beta = f0 / eta

        # accurate rectangular envelope
        majenv = self.__norminv(np.array([1-pf, 1+pf])/2, 0, f0/(np.sqrt(2)*np.pi*gamma))
        minenv = self.__norminv(np.array([1-pf, 1+pf])/2, 0, f0/(np.sqrt(2)*np.pi*eta))

        self.__compute_fhigh(majenv[1],minenv[1])

        envelope = self.__accurate_envelope_f(majenv[1],minenv[1])

        envelope[:2] = envelope[:2] * Nx
        envelope[-2:] = envelope[-2:] * Ny

        envelope = np.array([np.floor(envelope[0]), np.ceil(envelope[1]), np.floor(envelope[2]), np.ceil(envelope[3])])

        nx = np.arange(envelope[0], envelope[1] + 1, 1, dtype=int)
        ny = np.arange(envelope[2], envelope[3] + 1, 1, dtype=int)

        u = nx / Nx; #frequencies that bank contains
        v = ny / Ny;

        U, V = np.meshgrid(u,v)

        Uenvelope =  (np.add(-U * np.sin(theta), V * np.cos(theta)))**2 / beta**2
        Venvelope =  (np.add(U * np.cos(theta), V * np.sin(theta) - f0))**2 / alpha**2

        gf = np.exp( -(np.pi**2) * np.add(Uenvelope,Venvelope) )
        
        self.FrequencyDomainGaborFilter = gf
        self.envelope = envelope
        
        self.transform_from_frequency_to_spatial_domain(row, col)

    
    def show_parameters(self):
        print("This Gabor filter has: \n", "\t frequency:",self.frequency,
                                            "\n\t orientation:", self.orientation,
                                            "\n\t max frequency:", self.max_frequency,
                                            "\n\t Gaussian envelope:", self.envelope,                        
                                            "\n\t shape on frequency domain:", self.FrequencyDomainGaborFilter.shape,             
                                            "\n\t shape on spatial domain:", self.SpatialDomainGaborFilter.shape)
    
