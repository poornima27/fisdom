from PIL import Image
import numpy as np

# function to crop the image 
def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    print ymin,ymax,xmin,xmax
    return ymin,ymax,xmin,xmax
    #return img[ymin:ymax+1, xmin:xmax+1]

def Gray_image(img):
    """grayscale an image."""

    image = img.convert('L')  # convert image to monochrome
    image_to_convert = np.array(image)
    image_array = binarize_array(image_to_convert)
    return image_array
    

def binarize_array(numpy_array, threshold=127):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 0
            else:
                numpy_array[i][j] = 255
    return numpy_array


img = Image.open("data1.png").convert('RGB')
img_arr= np.array(img)
print img_arr.shape
bin_arr = Gray_image(img)

ymin,ymax,xmin,xmax = bbox2(bin_arr)
#text = binarize_array(text_inv)

for i in range(len(img_arr)):
    for j in range(len(img_arr[0])):
        if (img_arr[i][j][0] < 127 and img_arr[i][j][1] < 127 and img_arr[i][j][2] < 127):
            img_arr[i][j] = [0,0,128]

# result1 = Image.fromarray(img_arr)
# result1.show() 

text =  img_arr[ymin:ymax+1, xmin:xmax+1]     

result = Image.fromarray(text)
result.show()
result.save('data1_result_crop.jpg')
