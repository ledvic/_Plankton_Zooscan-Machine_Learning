import numpy as np
from skimage.util import img_as_float
import matplotlib.pyplot as plt


from .gabor_filter import GaborFilterBank as gbb

class GaborFilterResponse:
    # constructor
    def __init__(self, **kwargs):
        # private members
        self.__Zoom = None
        self.__ActualZoom = None
        self.__Method = None
        self.__Frequency = None # 
        self.__Response = None # 
        self.__ResponseSize = None # 

    @property # first decorate the getter method
    def zoom(self): # This getter method name is *the* name
        return self.__Zoom
    @zoom.setter    # the property decorates with `.setter` now
    def zoom(self, value):   # name, e.g. "attribute", is the same
        self.__Zoom = value   # the "value" name isn't special
    
    @property 
    def actual_zoom(self): 
        return self.__ActualZoom
    @actual_zoom.setter 
    def actual_zoom(self, value):   
        self.__ActualZoom = value
        
    @property 
    def FilterMethod(self): 
        return self.__Method
    @FilterMethod.setter 
    def FilterMethod(self, value):   
        self.__Method = value   
        
    @property 
    def frequency(self): 
        return self.__Frequency
    @frequency.setter 
    def frequency(self, value):   
        self.__Frequency = value   

    @property 
    def FilteredResponse(self): 
        return self.__Response
    @FilteredResponse.setter 
    def FilteredResponse(self, value):   
        self.__Response = value   

    @property 
    def SizeOfResponse(self): 
        return self.__ResponseSize
    @SizeOfResponse.setter 
    def SizeOfResponse(self, value):   
        self.__ResponseSize = value   
        
    def show_parameters(self):
        print("This Gabor filter has: \n", "\t frequency:",self.frequency,
                                            "\n\t zoon value:", self.zoom,
                                            "\n\t actual zoon value:", self.actual_zoom,
                                            "\n\t method of filtering:", self.FilterMethod,                        
                                            "\n\t shape on Filtered Response:", self.FilteredResponse.shape,             
                                            "\n\t shape of Filtered Response:", self.SizeOfResponse)
        
    def filter_an_image_with_a_set_of_Gabor_filters(self, FilteredImage, GaborFilterBank, FrequencyValue, NumberOfOrientations, arrMN, Method=0, FilterDomain=1):
                
        self.FilterMethod = Method

        if self.FilterMethod == 0:
            self.zoom = 1
            self.actual_zoom = [1, 1]
            self.SizeOfResponse = arrMN
            
            
        if FilterDomain == 1:
             
            self.frequency = FrequencyValue

            GaborFilterSubsetByFrequency = gbb().get_gabor_filters_by_frequency(GaborFilterBank, FrequencyValue, NumberOfOrientations)

            # zero memory for filter responses when filtering all points with original resolution
            if self.FilterMethod == 0:
                self.FilteredResponse = np.zeros((NumberOfOrientations, arrMN[1], arrMN[0]), dtype=np.complex_)

            for j in range(NumberOfOrientations):

                envelope = GaborFilterSubsetByFrequency[j].envelope

                # method 0 (full size responses)
                if self.FilterMethod == 0:

                    f2_ = np.zeros((arrMN[1] , arrMN[0]), dtype=np.complex_)

                    self.zoom = 1

                    lx = envelope[1] - envelope[0];
                    ly = envelope[3] - envelope[2];

                    # coordinates for the filter area in filtered fullsize image
                    xx = np.int32(np.mod( np.arange(0, lx + 1) + envelope[0] + arrMN[0] , arrMN[0] ) + 1) - 1;
                    yy = np.int32(np.mod( np.arange(0, ly + 1) + envelope[2] + arrMN[1] , arrMN[1] ) + 1) - 1;

                    fs = np.zeros((len(yy),len(xx)), dtype=np.complex_)

                    for y in range(len(yy)):
                        for x in range(len(xx)):
                            xf2 = xx[x] 
                            yf2 = yy[y] 
                            fs[y,x] = FilteredImage[yf2,xf2]


                    multifsgb = np.multiply(GaborFilterSubsetByFrequency[j].FrequencyDomainGaborFilter, fs)

                    for y in range(len(yy)):
                        for x in range(len(xx)):
                            xf2 = xx[x] 
                            yf2 = yy[y] 
                            f2_[yf2,xf2] = multifsgb[y,x]

                    self.FilteredResponse[j] = np.fft.ifftshift( np.fft.ifft2(f2_) )



class GaborFilteredResponseBank:

    # def __init__(self):
    #     self.__ListOfGaborFilterredResponses = None

    # @property # first decorate the getter method
    # def GaborFilteredResponseBank(self): # This getter method name is *the* name
    #     return self.__ListOfGaborFilterredResponses
    # @GaborFilteredResponseBank.setter    # the property decorates with `.setter` now
    # def GaborFilteredResponseBank(self, value):   # name, e.g. "attribute", is the same
    #     self.__ListOfGaborFilterredResponses = value   # the "value" name isn't special


    def create_a_set_of_Gabor_filtered_responses(self, Image, GaborFilterBank, Method=0, FilterDomain=1, MaxZoom=0):  
       
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
                GaborFilterResponseIJ = GaborFilterResponse()
                GaborFilterResponseIJ.filter_an_image_with_a_set_of_Gabor_filters(FilteredImage, GaborFilterBank, FrequencyValue, NumberOfOrientations, arrMN)

                ListOfGaborFilterredResponses[i] = GaborFilterResponseIJ

        return ListOfGaborFilterredResponses
    
    def convert_a_set_Gabor_filtered_responses_to_ndarray(self, ListOfGaborFilterredResponses, normalize = 1):
    
        NumberOfOrientations = ListOfGaborFilterredResponses[0].FilteredResponse.shape[0]
        NumberOfFrequencies = len(ListOfGaborFilterredResponses)

        n = ListOfGaborFilterredResponses[0].FilteredResponse.shape

        # handle case with responses from all points
        if len(n) == 3:

            meh = np.zeros((n[1],n[2], NumberOfFrequencies * NumberOfOrientations), dtype=np.complex_)
        
        for i in range(NumberOfFrequencies):
            for u in range(NumberOfOrientations):
                meh[:,:,i*NumberOfOrientations + u] = np.copy(ListOfGaborFilterredResponses[i].FilteredResponse[u,:,:])

        # if normalize > 0:
        #     mehTemporary = np.copy(meh)
        #     mehSub = np.einsum('kli->lik', 1.0 / np.tile(np.sqrt(np.sum(np.square(np.abs(mehTemporary)), axis=2)), [NumberOfFrequencies * NumberOfOrientations,1,1]))
        #     meh = np.multiply(mehSub, mehTemporary)
        if normalize > 0:
            meh = np.ascontiguousarray(meh)
            norms = np.sqrt(np.sum(np.square(np.abs(meh)), axis=2))
            for i in range(NumberOfFrequencies*NumberOfOrientations):
                meh[:,:,i] /= norms            
        return meh
    
    def display_image_with_its_responses(self, img, meh, NumberOfRows=8, FigSize=(15,10)):
    
        NumberOfRows = NumberOfRows
        
        fig, axes = plt.subplots(nrows=NumberOfRows, ncols=3, figsize=FigSize, layout="compressed")
        plt.gray()

        fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)

        # Plot original images
        for i, ax in zip(range(NumberOfRows), axes[:, 0]):
            ax.axis('off')
            if i == 0:
                ax.set_title("input image")
            if i == np.floor(NumberOfRows/2):
                ax.imshow(img, cmap='Greys_r')
                ax.axis('on')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            # ax.axis('off')
            
            # ax.yticks([])
            
        # Plot real part
        for i, ax in zip(range(NumberOfRows), axes[:, 1]):
            ax.imshow(np.real(meh[:,:,i]),cmap='Greys_r')
            if i == 0:
                ax.set_title("Real")
            ax.axis('off')

        # Plot imaginary part
        for i, ax in zip(range(NumberOfRows), axes[:, 2]):
            ax.imshow(np.imag(meh[:,:,i]),cmap='Greys_r')
            if i == 0:
                ax.set_title("Imaginary")
            ax.axis('off')

        fig.show()