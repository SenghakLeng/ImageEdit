import numpy as np 
import streamlit as st
import array
import io
from PIL import Image
from scipy.ndimage import median_filter
from scipy.ndimage import generic_filter
import cv2
# '''
# Each function (add, sub, add 2 img, histogram equalization) take the image in numpy arrray format
# and output the image also in numpy array format 

# read_raw take input in the form of file path

# '''

def read_raw(raw_image): # the input is the file path 
    img = raw_image.read()
    img_data = array.array('B', img)
    size = int(np.sqrt(len(img)))
    img_data = (np.array(img_data, dtype=np.uint8)).reshape(size, size)
    image = np.array(img_data, dtype= np.uint8)
    return image

def convert_type(img):
    if img.name[-3:] == 'raw':
        return read_raw(img)
    else:
        img_bytes = img.read()
        image_bytes = io.BytesIO(img_bytes)
        new_img = Image.open(image_bytes)
        return np.array(new_img)

def resize_image(image, new_size):
    height, width = image.shape[:2]
    new_height, new_width = new_size

    scale_y = new_height / height
    scale_x = new_width / width

    y = np.arange(new_height).reshape(-1, 1)
    x = np.arange(new_width).reshape(1, -1)

    y_orig = y / scale_y
    x_orig = x / scale_x

    y_int = np.floor(y_orig).astype(np.int32)
    x_int = np.floor(x_orig).astype(np.int32)
    y_frac = y_orig - y_int
    x_frac = x_orig - x_int

    y_int = np.clip(y_int, 0, height - 2)
    x_int = np.clip(x_int, 0, width - 2)

    top_left = image[y_int, x_int]
    top_right = image[y_int, x_int + 1]
    bottom_left = image[y_int + 1, x_int]
    bottom_right = image[y_int + 1, x_int + 1]

    top = top_left * (1 - x_frac) + top_right * x_frac
    bottom = bottom_left * (1 - x_frac) + bottom_right * x_frac

    resized_image = top * (1 - y_frac) + bottom * y_frac

    return resized_image.astype(image.dtype)

def convert_to_gray(image):
    if image.ndim > 2:
        img_gray = (0.2989 * image[:,:,0]) + (0.5870 * image[:,:,1]) +( 0.1140 * image[:,:,2])
        img_gray = np.array(img_gray, dtype= np.uint8)
    else: 
        img_gray = image
    return img_gray

def normalize_image(image):
    norm_img = np.clip(image, 0, 255)
    return norm_img


def add_sub_val_img(image, value):
    img_gray = convert_to_gray(image)
    new_img = img_gray.astype(np.int16) + value
    new_img_normalize = normalize_image(new_img).astype(np.uint8)
    return new_img_normalize

def hist_equalization(image):
    img_gray = convert_to_gray(image)
    hist, bins = np.histogram(img_gray.flatten(), bins=256, range=(0,255))
    cdf = hist.cumsum()
    cdf_normalized = (cdf *hist.max())/cdf.max()
    hist_eq_img = np.interp(img_gray.flatten(), bins[:-1], cdf_normalized).reshape(img_gray.shape)
    hist_eq_img = (hist_eq_img - hist_eq_img.min()) / (hist_eq_img.max() - hist_eq_img.min())
    hist_eq_img_norm = normalize_image(hist_eq_img)
    return hist_eq_img_norm

def add_sub_2_img(image1, image2, add):
    img_gray1 = convert_to_gray(image1)
    img_gray2 = convert_to_gray(image2)
    if add is True:
        new_img = img_gray1.astype(np.int16) + img_gray2.astype(np.int16)
    else:
        new_img = img_gray1.astype(np.int16) - img_gray2.astype(np.int16)
    new_img_norm = normalize_image(new_img).astype(np.uint8)
    return new_img_norm

def sharp_and_blur(image, mode):
    h, w = image.shape
    sharpen_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    average_filter = np.array([[1/25, 1/25, 1/25, 1/25, 1/25],
                               [1/25, 1/25, 1/25, 1/25, 1/25],
                               [1/25, 1/25, 1/25, 1/25, 1/25],
                               [1/25, 1/25, 1/25, 1/25, 1/25],
                               [1/25, 1/25, 1/25, 1/25, 1/25]])
    if mode == 'Sharpening':
        mod_img = np.convolve(image.reshape(-1), sharpen_filter.reshape(-1), mode = 'same').astype(np.int16)
    elif mode == 'Blurring':
        mod_img = np.convolve(image.reshape(-1), average_filter.reshape(-1), mode = 'same').astype(np.int16)
    new_img = normalize_image(mod_img.reshape(h, w)).astype(np.uint8)
    return new_img

def edge_detect(img):
    h, w =img.shape
    sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_y = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv_x = np.convolve(img.reshape(-1), sobel_x.reshape(-1), mode = 'same').astype(np.float32)
    conv_y = np.convolve(img.reshape(-1), sobel_y.reshape(-1), mode = 'same').astype(np.float32)
    magnitude = (np.power(conv_x, 2) + np.power(conv_y,2)).astype(np.float32)
    magnitude = np.sqrt(magnitude)
    new_img = magnitude.reshape(h, w)
    new_img = (new_img - new_img.min())/(new_img.max() - new_img.min())
    # new_img = normalize_image(new_img).astype(np.uint8)
    return new_img

def med_filter(img):
    new_img = median_filter(img,size=5)
    new_img = normalize_image(new_img)
    return new_img

def erosion_dilation(img, erosion, method, neighbor_size):
    kernel =  np.array([[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]], dtype=np.uint8)
    def minimum_filter(window):
        return np.min(window)
    def maximum_filter(window):
        return np.max(window)
    if erosion:
        if method == "min_filter":
            filtered_image = generic_filter(img, minimum_filter, size=neighbor_size)
        if method == "structuring element":
            filtered_image = cv2.erode(img, kernel, iterations=1)
    else:
        if method == "max_filter":
            filtered_image = generic_filter(img, maximum_filter, size=neighbor_size)
        if method == "structuring element":
            filtered_image = cv2.dilate(img, kernel=kernel, iterations=1)
    return normalize_image(filtered_image)


def count_objects(img):
    if len(img.shape) > 2:
        img = convert_to_gray(img)
        
    _, thres_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    thres_img[thres_img < 255] = 0
    thres_img[thres_img >=255] =1
    
    labeled_image, num_labels = label_objects(thres_img)

    object_count = num_labels - 1

    return object_count, labeled_image

def label_objects(binary_image):
    labeled_image = np.zeros_like(binary_image, dtype=np.uint32)
    current_label = 1

    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i, j] == 1:
                neighbors = get_neighbors(labeled_image, i, j)
                if not neighbors:
                    labeled_image[i, j] = current_label
                    current_label += 1
                else:
                    min_label = min(neighbors)
                    labeled_image[i, j] = min_label
                    propagate_equivalences(labeled_image, neighbors, min_label)

    num_labels = np.max(labeled_image)
    return labeled_image, num_labels

def get_neighbors(labeled_image, i, j):
    neighbors = []
    if i > 0 and labeled_image[i - 1, j] != 0:
        neighbors.append(labeled_image[i - 1, j])
    if j > 0 and labeled_image[i, j - 1] != 0:
        neighbors.append(labeled_image[i, j - 1])
    return neighbors

def propagate_equivalences(labeled_image, neighbors, min_label):
    for label in neighbors:
        if label != min_label:
            labeled_image[labeled_image == label] = min_label




def opening_closing(img, operation_iteration, opening):
    if opening:
        for i in range(operation_iteration):
            img = erosion_dilation(img = img, erosion=True, method= "structuring element",neighbor_size=5)
        for j in range(operation_iteration):
            img = erosion_dilation(img = img, erosion=False, method= "structuring element",neighbor_size=5)
        return img
    else: 
        for i in range(operation_iteration):
            img = erosion_dilation(img = img, erosion=False, method= "structuring element",neighbor_size=5)
        for j in range(operation_iteration):
            img = erosion_dilation(img = img, erosion=True, method= "structuring element",neighbor_size=5)
        return img
def main_page():
    st.title("Image Edit GUI")
    st.header("Welcome to Image Edit GUI")
    st.write('Created by: Leng Seng Hak (ID: 2023199007)')
    st.write('Course: Image Processing Class')
    st.write('Professor: Jae-Won Suh')
    options = ["Show Image", 
               "Adding or Subtracting Value to Image", 
               "Histogram Equalization", 
               "Adding 2 Images", 
               "Substracting 2 Images",
               "Sharpening or Blurring",
               "Detect Edges",
               "Apply Median Filtering",
               "Erosion and Dilation", 
               "Opening and Closing"]
    selected_option = st.selectbox("Select an option", options)
    st.write('You Selected: ', selected_option)
    return selected_option





def show_img_page():
    file_upload = st.file_uploader("Upload your image here")
    if file_upload is not None:
        image = convert_type(file_upload)
        st.image(image, caption ='This is the image you selected')

def add_sub_val_img_page():
    col1, col2 = st.columns(2)
    with col1:
        file_upload = st.file_uploader("Upload your image here...")
    with col2: 
        value = st.slider("Adjust the value", -255, 255, 0)

    ncol1, ncol2 = st.columns(2)
    with ncol1:
        if file_upload is not None:
            image  = convert_type(file_upload)
            st.image(image, caption= "This is the original image")
    with ncol2:
        if file_upload is not None:
            img_gray = convert_to_gray(image)
            new_img = add_sub_val_img(image=img_gray, value=value)
            st.image(new_img, caption = 'This is the modified image')
            
def hist_eq_page():
    file_upload = st.file_uploader("Upload your image here...")
    col1, col2 = st.columns(2)
    with col1:
        if file_upload is not None:
            image = convert_type(file_upload)
            st.image(image, caption= "This is the original image")
    with col2:
        if file_upload is not None:
            new_img = hist_equalization(image= image)
            new_img = resize_image(new_img, (image.shape[0], image.shape[1]))
            st.image(new_img, caption = 'Image after apply histogram equalization')

def add_sub_2img_page(is_add):
    col1, col2 = st.columns(2)
    with col1:
        file_1 = st.file_uploader("Upload your first image here...")
    with col2: 
        file_2 = st.file_uploader("Uploade your second image here...")

    ncol1, ncol2, ncol3 = st.columns(3)
    with ncol1: 
        if file_1 is not None:
            image1 = convert_type(file_1)
            st.image(image1, caption = "This is the first original image")
    with ncol2:
        if file_2 is not None:
            image2 = convert_type(file_2)
            image2 = resize_image(image2, (image1.shape[0], image1.shape[1]))
            st.image(image2, caption = "This is the second original image")
    with ncol3: 
        if (file_1 is not None) and (file_2 is not None):
            new_img = add_sub_2_img(image1=image1, image2= image2, add= is_add)
            st.image(new_img, caption= "This is the new image")

def sharp_blurr_page():
    col1, col2 = st.columns(2)
    with col1:
        file_upload = st.file_uploader("Upload your image here...")
    with col2: 
        mode = st.radio("Choose your option", ("Sharpening", "Blurring"))

    ncol1, ncol2 = st.columns(2)
    with ncol1:
        if file_upload is not None:
            image  = convert_type(file_upload)
            st.image(image, caption= "This is the original image")
    with ncol2:
        if file_upload is not None:
            img_gray = convert_to_gray(image)
            new_img = sharp_and_blur(image=img_gray, mode=mode) 
            st.image(new_img, caption = 'This is the modified image')

def edge_detect_page():
    file_upload = st.file_uploader("Upload your image here...")
    col1, col2 = st.columns(2)
    with col1:
        if file_upload is not None:
            image = convert_type(file_upload)
            st.image(image, caption= "This is the original image")
    with col2:
        if file_upload is not None:
            img_gray = convert_to_gray(image)
            new_img = edge_detect(img_gray)
            st.image(new_img, caption = 'This is the Edge Detected from the Image')

def med_filter_page():
    file_upload = st.file_uploader("Upload your image here...")
    col1, col2 = st.columns(2)
    with col1:
        if file_upload is not None:
            image = convert_type(file_upload)
            st.image(image, caption= "This is the original image")
    with col2:
        if file_upload is not None:
            img_gray = convert_to_gray(image)
            new_img = med_filter(img_gray)
            st.image(new_img, caption = 'This is the Image after Applying Median Filter')

def erosion_dilation_page():
    file_upload = st.file_uploader("Upload your image here...")

    if file_upload is not None:
        # Load the image
        image = convert_type(file_upload)
        # Erosion and Dilation tabs
        tab1, tab2 = st.tabs(['Erosion', 'Dilation'])

        with tab1:
            method = st.radio("Which method you want to use in the erosion?", ["Minimum Filter", "Structuring Element"])
            img_gray = convert_to_gray(image)
            neighbor_size = 5
            t1col = st.columns(2)
            with t1col[0]:
                st.image(image, caption="This is the original image")
            with t1col[1]:
                if method == "Minimum Filter":
                    new_img = erosion_dilation(img=img_gray, erosion=True, method="min_filter", neighbor_size=neighbor_size)
                    st.image(new_img, caption='This is the Image after Applying Erosion using Minimum Filter')

                if method == "Structuring Element":
                    new_img = erosion_dilation(img=img_gray, erosion=True, method="structuring element", neighbor_size=neighbor_size)
                    st.image(new_img, caption='This is the Image after Applying Erosion using Structuring Element')

        with tab2:
            method = st.radio("Which method you want to use in the dilation?", ["Maximum Filter", "Structuring Element"])
            t2col = st.columns(2)
            with t2col[0]:
                st.image(image, caption="This is the original image")
            with t2col[1]:
                if method == "Maximum Filter":
                    new_img = erosion_dilation(img=img_gray, erosion=False, method="max_filter", neighbor_size=neighbor_size)
                    st.image(new_img, caption='This is the Image after Applying Dilation using Maximum Filter')

                if method == "Structuring Element":
                    new_img = erosion_dilation(img=img_gray, erosion=False, method="structuring element", neighbor_size=neighbor_size)
                    st.image(new_img, caption='This is the Image after Applying Dilation using Structuring Element')

def opening_closing_page():
    file_upload = st.file_uploader("Upload your image here...")

    if file_upload is not None:
        image = convert_type(file_upload)
        num_operation = st.slider('Selected Number of Itreation', min_value=1, max_value=12)
        tab1, tab2 = st.tabs(['Opening', 'Closing'])


        with tab1:
            img_gray = convert_to_gray(image)
            t1col = st.columns(2)
            with t1col[0]:
                st.image(image, caption="This is the original image")
            with t1col[1]:
                new_img = opening_closing(img = image, opening=True, operation_iteration=num_operation)
                st.image(new_img, caption=f'This is the Image after opening for {num_operation} iterations')
            object_count, _ = count_objects(new_img)
            st.markdown(f"<h1 style='text-align: center; font-size: 30px; \
                        '>In this image there are {object_count} objects</h1>", 
                        unsafe_allow_html=True)
                
        with tab2:
            t2col = st.columns(2)
            with t2col[0]:
                st.image(image, caption="This is the original image")
            with t2col[1]:
                new_img = opening_closing(img = image, opening=False, operation_iteration=num_operation)
                st.image(new_img, caption=f'This is the Image after closing for {num_operation} iterations')
            object_count, _ = count_objects(new_img)
            st.markdown(f"<h1 style='text-align: center; font-size: 30px; \
                        '>In this image there are {object_count} objects</h1>", 
                        unsafe_allow_html=True)


def main(): 
    option = main_page()
    if option == "Show Image":
        show_img_page()
    elif option == "Adding or Subtracting Value to Image":
        add_sub_val_img_page()
    elif option == "Histogram Equalization":
        hist_eq_page()
    elif option == "Adding 2 Images":
        add_sub_2img_page(is_add= True)
    elif option == "Substracting 2 Images":
        add_sub_2img_page(is_add=False)
    elif option =="Sharpening or Blurring":
        sharp_blurr_page()
    elif option =="Detect Edges":
        edge_detect_page()
    elif option == "Apply Median Filtering":
        med_filter_page()
    elif option == "Erosion and Dilation":
        erosion_dilation_page()
    elif option =="Opening and Closing":
        opening_closing_page()

    
if __name__ == "__main__":
    main()
