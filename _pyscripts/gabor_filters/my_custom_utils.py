import numpy as np
# __all__ = ["gabor_filer"]
import gabor_filter 
import gabor_filter_response
from skimage.util import img_as_float

import matplotlib.pyplot as plt


def get_gabor_filters_by_frequency(GaborFilterBank, FrequencyValue, NumberOfOrientations):

    GaborFilterSubsetByFrequency = [None]*NumberOfOrientations
    
    j=0
    
    for i in range(len(GaborFilterBank)):
        if GaborFilterBank[i].frequency == FrequencyValue:
            GaborFilterSubsetByFrequency[j] = GaborFilterBank[i]
            j += 1
            if j == NumberOfOrientations:
                return GaborFilterSubsetByFrequency
            
    return GaborFilterSubsetByFrequency

def get_max_frequency_of_Gabor_filter_bank(GaborFilterBank):
       
    MaxFrequency = 0
    
    for i in range(len(GaborFilterBank)):
        if GaborFilterBank[i].max_frequency > MaxFrequency:
            MaxFrequency = GaborFilterBank[i].max_frequency
            
    return MaxFrequency



def solve_k(gamma, p):

        x = 1.0 / (gamma * np.pi)*np.sqrt(-np.log(p));
        k = (1 + x) / (1 - x);

        return k


def solve_p(gamma, k):

    p = np.exp(- (gamma * np.pi * (k - 1) / (k + 1))**2);

    return p


def solve_gamma(k, p):

    gamma = 1.0 / np.pi * np.sqrt(-np.log(p)) * (k + 1) / (k - 1);

    return gamma


def solve_eta(n, p):

    ua = np.pi / n / 2; # ua based on approximation

    eta = 1.0 / np.pi * np.sqrt(-np.log(p)) / ua;

    return eta

def solve_filter_parameters(k, p, m, n):

    gamma = solve_gamma(k, p);
    eta = solve_eta(n, p);

    return gamma, eta


def create_a_set_of_gabor_filters(fmax=0.327, k=np.sqrt(2), p=0.5, u=6, v=8, row=43, col=43):
#     row=43, col=43 # size of image
#     fmax = 0.327 # maximum frequency

#     k = np.sqrt(2) #frequency ratio or factor for selecting filter frequencies

#     p = 0.5 # crossing point between two consecutive filters, default 0.5
#     # pf = np.sqrt(0.99) #energy to include in the filters

#     u = 6 #number of frequencies
#     v = 8 #number of orientation

    gamma, eta = solve_filter_parameters(k, p, u, v) # smoothing parameters

    GaborFilterBank = [None]*u*v 
      
    for i in range(0,u):
        fu = fmax/k**i # frequency of the filter
        for j in range(0,v):
            theta = j/v*np.pi; # orientation of the filter 

            GaborFilterIJ = gabor_filter.GaborFilter()
            GaborFilterIJ.frequency = fu
            GaborFilterIJ.orientation = theta
            GaborFilterIJ.compute_a_gabor_filter(gamma, eta, row, col) 

            # GaborFilterIJ.showParameters()

            GaborFilterBank[i*v+j] = GaborFilterIJ
    
    return GaborFilterBank


def get_Gabor_filters_by_frequency(GaborFilterBank, FrequencyValue, NumberOfOrientations):

    GaborFilterSubsetByFrequency = [None]*NumberOfOrientations
    
    j=0
    
    for i in range(len(GaborFilterBank)):
        if GaborFilterBank[i].frequency == FrequencyValue:
            GaborFilterSubsetByFrequency[j] = GaborFilterBank[i]
            j += 1
            if j == NumberOfOrientations:
                return GaborFilterSubsetByFrequency
            
    return GaborFilterSubsetByFrequency

def get_max_frequency_of_Gabor_filter_bank(GaborFilterBank):
       
    MaxFrequency = 0
    
    for i in range(len(GaborFilterBank)):
        if GaborFilterBank[i].max_frequency > MaxFrequency:
            MaxFrequency = GaborFilterBank[i].max_frequency
            
    return MaxFrequency


def create_a_set_of_Gabor_filtered_responses(Image, GaborFilterBank, Method=0, FilterDomain=1, MaxZoom=0):  
       
    if FilterDomain == 1:
        # perform the filtering
        FilteredImage = np.fft.fft2(np.fft.ifftshift(img_as_float(Image)))

        #the loop for calculating responses at all frequencies

        # Get all values of frequencies from bank of Gabor filters
        ListOfrequencies = np.zeros(len(GaborFilterBank))
        for i in range(len(GaborFilterBank)):
            ListOfrequencies[i] = GaborFilterBank[i].frequency
        ListOfrequencies = np.unique(ListOfrequencies)
        ListOfrequencies[::-1].sort()

        NumberOfOrientations = np.int32(len(GaborFilterBank) / len(ListOfrequencies[::-1]))

        ListOfGaborFilterredResponses = [None]*len(ListOfrequencies)
        
        arrMN = np.array([Image.shape[1], Image.shape[0]])
            
        for i in range(len(ListOfrequencies)):   
            FrequencyValue = ListOfrequencies[i]
            GaborFilterResponseIJ = gabor_filter_response.GaborFilterResponse()
            GaborFilterResponseIJ.filter_an_image_with_a_set_of_Gabor_filters(FilteredImage, FrequencyValue, NumberOfOrientations, arrMN)

            ListOfGaborFilterredResponses[i] = GaborFilterResponseIJ

    return ListOfGaborFilterredResponses


def convert_a_set_Gabor_filtered_responses_to_ndarray(ListOfGaborFilterredResponses, normalize = 1):
    
    NumberOfOrientations = ListOfGaborFilterredResponses[0].FilteredResponse.shape[0]
    NumberOfFrequencies = len(ListOfGaborFilterredResponses)

    n = ListOfGaborFilterredResponses[0].FilteredResponse.shape

    # handle case with responses from all points
    if len(n) == 3:

        meh = np.zeros((n[1],n[2], NumberOfFrequencies * NumberOfOrientations), dtype=np.complex_)
    
    for i in range(NumberOfFrequencies):
        for u in range(NumberOfOrientations):
            meh[:,:,i*NumberOfOrientations + u] = np.copy(ListOfGaborFilterredResponses[i].FilteredResponse[u,:,:])

    if normalize > 0:
        mehTemporary = np.copy(meh)
        mehSub = np.einsum('kli->lik', 1.0 / np.tile(np.sqrt(np.sum(np.square(np.abs(mehTemporary)), axis=2)), [NumberOfFrequencies * NumberOfOrientations,1,1]))
        meh = np.multiply(mehSub, mehTemporary)
        
    return meh


def display_Gabor_filter_bank_on_spatial_domain(GaborFilterBank, NumberOfFrequencies, NumberOfOrientations, FigSize=(15.10)):
    plt.figure(FigSize) 
    for i in range(len(GaborFilterBank)):
        ax = plt.subplot(NumberOfFrequencies, NumberOfOrientations, i + 1)
        ax.imshow(np.real(GaborFilterBank[i].SpatialDomainGaborFilter),cmap='Greys_r')
        label = str(np.real(GaborFilterBank[i].SpatialDomainGaborFilter).shape)
        ax.set_title(label)
        ax.axis("off")
  
        
def display_Gabor_filter_bank_on_frequency_domain(GaborFilterBank, NumberOfFrequencies, NumberOfOrientations,FigSize=(15.10) ):
    plt.figure(FigSize) 
    for i in range(len(GaborFilterBank)):
        ax = plt.subplot(NumberOfFrequencies, NumberOfOrientations, i + 1)
        ax.imshow(np.real(GaborFilterBank[i].FrequencyDomainGaborFilter),cmap='Greys_r')
        label = str(np.real(GaborFilterBank[i].FrequencyDomainGaborFilter).shape)
        ax.set_title(label)
        ax.axis("off")
        
        
def display_image_with_its_responses(img, meh, FigSize=(15.10)):
    
    NumberOfRows = meh.shape[2]
    
    fig, axes = plt.subplots(nrows=NumberOfRows, ncols=3, figsize=FigSize)
    plt.gray()

    fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)

    # Plot original images
    for i, ax in zip(range(NumberOfRows), axes[:, 0]):
        ax.imshow(img, cmap='Greys_r')
        if i == 0:
            ax.set_title("input image")
        ax.axis('off')

    for i, ax in zip(range(NumberOfRows), axes[:, 1]):
        ax.imshow(np.real(meh[:,:,i]),cmap='Greys_r')
        if i == 0:
            ax.set_title("Real")
        ax.axis('off')

    for i, ax in zip(range(NumberOfRows), axes[:, 2]):
        ax.imshow(np.imag(meh[:,:,i]),cmap='Greys_r')
        if i == 0:
            ax.set_title("Imaginary")
        ax.axis('off')

    fig.tight_layout()

    plt.show()
    
    
