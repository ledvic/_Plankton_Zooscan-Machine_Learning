import numpy as np

# __all__ = ["gabor_filter", "my_custom_utils"]
from my_custom_utils import get_gabor_filters_by_frequency

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

            GaborFilterSubsetByFrequency = get_gabor_filters_by_frequency(GaborFilterBank, FrequencyValue, NumberOfOrientations)

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


