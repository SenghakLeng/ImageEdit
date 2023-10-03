import numpy as np 
import streamlit as st
import array
import io
from PIL import Image

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
    norm_img = np.array((image - image.min()) / (image.max() - image.min()))
    return norm_img


def add_sub_val_img(image, value, add):
    img_gray = convert_to_gray(image)
    if add is True:
        new_img = img_gray + value
    else: 
        new_img = img_gray - value
    new_img_normalize = normalize_image(new_img)
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
        new_img = img_gray1 + img_gray2
    else:
        new_img = img_gray1 - img_gray2
    new_img_norm = normalize_image(new_img)
    return new_img_norm

def main_page():
    st.title("Image Edit GUI")
    st.header("Welcome to Image Edit GUI")
    st.write('Created by: Leng Seng Hak (ID: 2023199007)')
    st.write('Course: Image Processing Class')
    st.write('Professor: Jae-Won Suh')
    options = ["Show Image", 
               "Adding Value to Image", 
               "Substracting Value from Image", 
               "Histogram Equalization", 
               "Adding 2 Images", 
               "Substracting 2 Images"]
    selected_option = st.selectbox("Select an option", options)
    st.write('You Selected: ', selected_option)
    return selected_option

def show_img_page():
    file_upload = st.file_uploader("Upload your image here")
    if file_upload is not None:
        image = convert_type(file_upload)
        st.image(image, caption ='This is the image you selected')

def add_sub_val_img_page(is_add):
    col1, col2 = st.columns(2)
    with col1:
        file_upload = st.file_uploader("Upload your image here...")
    with col2: 
        value = st.number_input("Enter a number between 0-255", value=0, step=1)

    ncol1, ncol2 = st.columns(2)
    with ncol1:
        if file_upload is not None:
            image  = convert_type(file_upload)
            st.image(image, caption= "This is the original image")
    with ncol2:
        if file_upload is not None:
            img_gray = convert_to_gray(image)
            new_img = add_sub_val_img(image=img_gray, value=value, add = is_add)
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


def main(): 
    option = main_page()
    if option == "Show Image":
        show_img_page()
    elif option == "Adding Value to Image":
        add_sub_val_img_page(is_add= True)
    elif option == "Substracting Value from Image":
        add_sub_val_img_page(is_add=False)
    elif option == "Histogram Equalization":
        hist_eq_page()
    elif option == "Adding 2 Images":
        add_sub_2img_page(is_add= True)
    else:
        add_sub_2img_page(is_add=False)
    

main()
