import cv2

class BlurDetection ():
    def __init__(self, thresh):
        self.thresh = thresh

    def log_var(self, img):
        '''
        Tutorial from https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
        LoG operator returns edges, so low variance means more blurred due to fewer edges
        :param img: image to process (np
        :return:
        '''

        return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

    def fft_filter(self, img):
