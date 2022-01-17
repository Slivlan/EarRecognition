import cv2

class Preprocess:

    def histogram_equlization_rgb(self, img):
        # Simple preprocessing using histogram equalization 
        # https://en.wikipedia.org/wiki/Histogram_equalization

        intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)

        # For Grayscale this would be enough:
        # img = cv2.equalizeHist(img)

        return img

    # Add your own preprocessing techniques here.
    def edge_augmentation(self, img):
        img = image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #to grayscale
        imageBlurred = cv2.GaussianBlur(img, (3, 3), 0)
        imageBlurredEdges = cv2.Canny(imageBlurred, 10, 200)
        imgEdgesBlurred = cv2.GaussianBlur(imageBlurredEdges, (9,9), 3, borderType=cv2.BORDER_REPLICATE)

        img = cv2.subtract(img, cv2.multiply(imgEdgesBlurred, 0.5))

        return img