# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:24:26 2019

@author: Vit Saidl
"""

import shutil
import tkinter as tk
import tkinter.filedialog as fld
from tkinter import ttk
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
import numpy as np
import cv2
from PIL import Image, ImageTk
import tensorflow
from typing import List, Dict


class PredictedObject:
    """Class used as container for information about figure content.

    Args:
        order_number(int): Order number of probable classes from the most probable one
        (it has 1).
        class_name(str): Name of figure class (e.g. penquin).
        probability(float): Probability that the object on figure belongs to the class.
        class_index(int): Index of the class in a list of classes defined within ML model.
    """

    def __init__(
        self, order_number: int, class_name: str, probability: float, class_index: int
    ):
        self.order_number = order_number
        self.class_name = class_name
        self.probability = probability
        self.class_index = class_index


def load_image(file_name: str) -> np.ndarray:
    """Loads image and transforms it into the form in which neural net can work with it.

    Image is loaded with both heigh and width 224 px. Then the image is converted
    to numpy.ndarray with shape (224,224,3) and quasi-batch (1,224,224,3).
    Finally processing (zero-centering of channels) takes place.

    Args:
        file_name (str): File name including dir, e.g. C:\\images\\penquin.jpg.

    Returns:
        numpy.ndarray: Processed image in a tensor form.
    """
    original_image = image.load_img(file_name, target_size=(224, 224))
    image_array = image.img_to_array(original_image)
    image_batch = np.expand_dims(image_array, axis=0)
    processed_image = preprocess_input(image_batch)
    return processed_image


def load_image_into_window(
    labelframe_orig_figure: ttk.Labelframe,
    button_find_class: ttk.Button,
    button_create_heatmap: ttk.Button,
    button_save_figure: ttk.Button,
):
    """Shows figure choosen by user in window interface + sets
    global variable with figure name

    Args:
        labelframe_orig_figure(ttk.Labelframe): Container for original figure.
        button_find_class(ttk.Button): Find classes button.
        button_create_heatmap(ttk.Button): Create heatmap button.
        button_save_figure(ttk.Button): Save button.
    """
    global global_figure_name
    global_figure_name = fld.askopenfilename(
        initialdir=".",
        title="Select picture for classification",
        filetypes=[("jpeg files", "*.jpg")],
    )
    load_image_to_labelframe(global_figure_name, labelframe_orig_figure)
    button_find_class.config(state=tk.NORMAL)
    button_create_heatmap.config(state=tk.DISABLED)
    button_save_figure.config(state=tk.DISABLED)


def _get_class_index(prediction: np.ndarray, order_number_minus_one: int) -> int:
    """Returns model specified index of a class determined by order_number_minus_one.

    Args:
        prediction(np.ndarray): Probabilities for all classes.
        order_number_minus_one(int): Order number specifying class.

    Returns(int): Index of class specified by order_number_minus_one.
    """
    return np.where(
        prediction
        == np.partition(prediction.flatten(), -2)[-order_number_minus_one - 1]
    )[1][0]


def get_prediction_list(original_image: np.ndarray, model) -> List[PredictedObject]:
    """Determines ten most probable image classes.

    Determines 10 most probable classes of image. This info is afterwards processed
    into user-friendly class PredictedObject (more precisely into list with objects
    of this type).

    Args:
        original_image (np.ndarray): Figure - tensor, which we try to classify.
        model (tf.keras.engine.functional.Functional): used ML model.

    Returns:
        List[PredictedObject]: List of PredictedObject objects.
    """
    prediction = model.predict(original_image)
    ten_best_classes = decode_predictions(prediction, top=10)[0]
    predictions_list = []
    for order_number_minus_one, element in enumerate(ten_best_classes):
        class_name = element[1]
        probability = round(100 * element[2], 2)
        class_index = _get_class_index(prediction, order_number_minus_one)
        predictions_list.append(
            PredictedObject(
                order_number_minus_one + 1, class_name, probability, class_index
            )
        )
    return predictions_list


def determine_classes(
    file_name: str,
    model,
    textbox_classes_list: tk.Text,
    combobox_chosen_class: ttk.Combobox,
    button_save_figure: ttk.Button,
    button_create_heatmap: ttk.Button,
):
    """Fills combobox and text widget with figure classes prediction.

    Args:
        file_name (str): File name including dir, e.g. C:\\images\\penquin.jpg.
        model (keras.engine.training.Model): Used ML model.
        textbox_classes_list(tk.Text): List of predictions textbox.
        combobox_chosen_class(ttk.Combobox): Choose class combobox.
        button_save_figure(ttk.Button): Save image button.
        button_create_heatmap(ttk.Button): Show heatmap button.
    """
    figure = load_image(file_name)
    prediction_list = get_prediction_list(figure, model)
    classes_for_combobox = []
    global global_mapping_name_to_index
    textbox_classes_list.config(state=tk.NORMAL)
    textbox_classes_list.delete(1.0, tk.END)

    for figure_class in prediction_list:
        text_message = (
            f"{figure_class.order_number}. "
            f"class {figure_class.class_name} "
            f"with probability {figure_class.probability}%\n"
        )
        textbox_classes_list.insert(tk.END, text_message)
        classes_for_combobox.append(figure_class.class_name)
        global_mapping_name_to_index[figure_class.class_name] = figure_class.class_index

    textbox_classes_list.config(state=tk.DISABLED)
    combobox_chosen_class["values"] = classes_for_combobox
    button_save_figure.config(state=tk.DISABLED)
    button_create_heatmap.config(state=tk.NORMAL)


def create_feature_map(
    model, final_layer_name: str, class_index: int, processed_image: np.ndarray
) -> np.ndarray:
    """Returns last convolution layer multiplied by its importance for a given class.

    Args:
        model (keras.engine.training.Model): Used ML model.
        final_layer_name (str): Name of final convolution layer in ML model.
        class_index (int): Index of image class.
        processed_image (numpy.ndarray): Processed (convertion to numpy array) loaded figure.

    Returns:
        numpy.ndarray: Feature map with info about importance of channels for image class.
        Usually the tensor shape is (14, 14, 512).
    """
    class_output = model.output[:, class_index]
    last_conv_layer = model.get_layer(final_layer_name)
    filter_count = last_conv_layer.filters
    # gradients class_output for layer last_conv_layer
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    # average of gradients over all axes except the channel one
    # i.e. result consists of 512 elemets (number given by ML model layer)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    # function provides access to the above mentioned things
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([processed_image])
    # for each channel is conv_layer_output_value multiplied by channel importance
    # for a given image class
    for i in range(filter_count):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    return conv_layer_output_value


def create_heatmap(feature_map: np.ndarray) -> np.ndarray:
    """Returns heatmap - normalized average of channels of feature map.

    Args:
        feature_map (numpy.ndarray): Usually tensor with shape (14, 14, 512) and float type.

    Returns
        numpy.ndarray: Float tensor with shape (14,14) with values from interval <0,1>.
    """
    heatmap = np.mean(feature_map, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap)
    return heatmap


def join_heatmap_and_fig(file_name: str, heatmap: np.ndarray) -> np.ndarray:
    """Loads original image and puts over it the heatmap.

    Original image is loaded from disk, i.e. it has original shape, not (224, 224).

    Args:
        file_name (str): File name including dir, e.g. C:\\images\\penquin.jpg.
        heatmap (numpy.ndarray): Heatmap with shape (14,14) showing
        the most important parth of an image for a given image class.

    Returns:
        numpy.ndarray: Original image with heatmap overlay.
    """
    original_image = cv2.imread(file_name)
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap * 0.4 + original_image


def load_image_to_labelframe(file_name: str, labelframe_name: ttk.Labelframe):
    """Loads image from disk and puts it into chosen labelframe.

    Note: without double usage of converted_image loaded figure would not be shown.
    Args:
        file_name (str): File name including dir, e.g. C:\\images\\penquin.jpg.
        labelframe_name (tkinter.ttk.Labelframe): Object in which the image will be located.
    """
    loaded_image = Image.open(file_name)
    loaded_image = loaded_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
    converted_image = ImageTk.PhotoImage(loaded_image)
    fig_in_labelframe = ttk.Label(labelframe_name, image=converted_image)
    fig_in_labelframe.image = converted_image
    fig_in_labelframe.grid(column=0, row=0)


def create_final_figure(
    original_file_name: str,
    model,
    final_layer_name: str,
    mapping_name_to_index: Dict[str, int],
    combobox_chosen_class: ttk.Combobox,
    labelframe_fig_with_heatmap: ttk.Labelframe,
    button_save_figure: ttk.Button,
):
    """Creates heatmap overlaid over original figure and saves it to temporary file
    and window interface.

    Args:
        original_file_name (str): File name including dir, e.g. C:\\images\\penquin.jpg.
        model (keras.engine.training.Model): Used ML model.
        final_layer_name (str): Name of final convolution layer in ML model.
        mapping_name_to_index (Dict[str,int]): Mapping of class name to class index.
        combobox_chosen_class(ttk.Combobox): Choose class combobox.
        labelframe_fig_with_heatmap(ttk.Labelframe): Container for heatmap adjusted figure.
        button_save_figure(ttk.Button): Save image button.
    """
    chosen_class = combobox_chosen_class.get()
    index = mapping_name_to_index[chosen_class]

    processed_image = load_image(original_file_name)

    feature_map = create_feature_map(model, final_layer_name, index, processed_image)
    heatmap = create_heatmap(feature_map)
    heatmap_overlay = join_heatmap_and_fig(original_file_name, heatmap)
    cv2.imwrite("temp\\temporary_picture.jpg", heatmap_overlay)
    load_image_to_labelframe("temp\\temporary_picture.jpg", labelframe_fig_with_heatmap)
    button_save_figure.config(state=tk.NORMAL)


def save_final_image():
    """Takes copy of image from temp dir and saves it under the name specified by the user."""
    final_file_name = fld.asksaveasfilename(
        initialdir=".", title="Select file", filetypes=[("jpeg files", "*.jpg")]
    )
    final_file_name = final_file_name + ".jpg"
    shutil.copyfile("temp\\temporary_picture.jpg", final_file_name)


def create_window():
    """Creates window interface for application."""
    root = tk.Tk()
    root.title("Image recognition")
    mainframe = ttk.Frame(root, padding=(3, 3, 12, 12))
    mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
    mainframe.columnconfigure(0, weight=1)
    mainframe.rowconfigure(0, weight=1)

    labelframe_orig_figure = ttk.Labelframe(
        mainframe, text="Original image", width=IMAGE_WIDTH, height=IMAGE_HEIGHT
    )
    labelframe_orig_figure.grid_propagate(False)
    labelframe_orig_figure.grid(column=0, row=0, padx=10, pady=5)

    labelframe_fig_with_heatmap = ttk.Labelframe(
        mainframe, text="Image with heatmap", width=IMAGE_WIDTH, height=IMAGE_HEIGHT
    )
    labelframe_fig_with_heatmap.grid_propagate(False)
    labelframe_fig_with_heatmap.grid(column=1, row=0, padx=10, pady=5)

    button_load_image = ttk.Button(
        mainframe,
        text="Load image",
        command=lambda: load_image_into_window(
            labelframe_orig_figure,
            button_find_class,
            button_create_heatmap,
            button_save_figure,
        ),
    )
    button_load_image.grid(column=0, row=1)

    button_save_figure = ttk.Button(
        mainframe, text="Save image", command=save_final_image, state=tk.DISABLED
    )
    button_save_figure.grid(column=1, row=1)

    button_find_class = ttk.Button(
        mainframe,
        text="Determine the most probable class",
        command=lambda: determine_classes(
            global_figure_name,
            USED_MODEL,
            textbox_classes_list,
            combobox_chosen_class,
            button_save_figure,
            button_create_heatmap,
        ),
        state=tk.DISABLED,
    )
    button_find_class.grid(column=0, row=2, columnspan=2)

    textbox_classes_list = tk.Text(mainframe, height=10, width=50, state=tk.DISABLED)
    textbox_classes_list.grid(column=0, row=3, rowspan=3)

    label_class_choice = ttk.Label(mainframe, text="Choose class")
    label_class_choice.grid(column=1, row=3)

    combobox_chosen_class = ttk.Combobox(mainframe, state="readonly")
    combobox_chosen_class.grid(column=1, row=4)

    button_create_heatmap = ttk.Button(
        mainframe,
        text="Show heatmap for given class",
        command=lambda: create_final_figure(
            global_figure_name,
            USED_MODEL,
            FINAL_CONV_LAYER_NAME,
            global_mapping_name_to_index,
            combobox_chosen_class,
            labelframe_fig_with_heatmap,
            button_save_figure,
        ),
        state=tk.DISABLED,
    )
    button_create_heatmap.grid(column=1, row=5)

    root.mainloop()


tensorflow.compat.v1.disable_eager_execution()
IMAGE_HEIGHT = 500
IMAGE_WIDTH = 400
USED_MODEL = VGG19(weights="imagenet")
FINAL_CONV_LAYER_NAME = "block5_conv4"
global_figure_name = ""
global_mapping_name_to_index = {}

if __name__ == "__main__":
    create_window()
