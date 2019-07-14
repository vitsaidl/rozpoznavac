# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:24:26 2019

@author: Vit Saidl
"""

import shutil
import tkinter as tk
import tkinter.filedialog as fld
from tkinter import ttk
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras import backend as K
import numpy as np
import cv2
from PIL import Image, ImageTk

class PredikovanyObjekt:
    """Třída sloužící jako kontejner pro info spojené s třídou obrázku
    """
    def __init__(self, poradi, objekt, pravdepodobnost, index):
        self.poradi = poradi
        self.objekt = objekt
        self.pravdepodobnost = pravdepodobnost
        self.index = index

def nacti_obrazek(jmeno_souboru):
    """Zpracování obrázku do podoby, kterou dokáže síť zpracovat

    Obrázek uživatelem vybraný se načte. Jeho rozměry se změní na 224x224 px.
    Následně se objekt coby třída PIL.Image.Image převede na numpy.ndarray, tj.
    objekt nabyde tvaru (224, 224, 3). Posléze
    se převede na kvazibatch, tj. tenzor obrázku tvaru (224, 224, 3) se převede
    na (1, 224, 224, 3). Nakonec proběhne zprocesování pro vgg16 (normalizace
    na channely)

    Args:
        jmeno_souboru (string): Jméno zdrojového obrázku i s celou cestou

    Returns:
        numpy.ndarray: Zprocesovaný obrázek ve formě tenzoru
    """

    zdrojovy_obrazek = image.load_img(jmeno_souboru, target_size=(224, 224))
    obrazek_na_array = image.img_to_array(zdrojovy_obrazek)
    obrazek_batch = np.expand_dims(obrazek_na_array, axis=0)
    zprocesovany_obrazek = preprocess_input(obrazek_batch)
    return zprocesovany_obrazek

def dej_obrazek_do_okna(*args):
    """Uživatelem vybraný soubor je v programu zobrazen + se nastaví glob. prom. s jeho názvem
    """
    global global_jmeno_obrazku
    global_jmeno_obrazku = fld.askopenfilename(initialdir=".",
                                               title="Select picture for classification",
                                               filetypes=[("jpeg files", "*.jpg")])
    nacti_obrazek_do_labelframu(global_jmeno_obrazku, labelframe_puvodni_obrazek)
    tlacitko_zjisti_tridu.config(state=tk.NORMAL)
    tlacitko_vyrob_heatmapu.config(state=tk.DISABLED)
    tlacitko_uloz_obrazek.config(state=tk.DISABLED)

def ziskej_list_predikci(zdrojovy_obrazek, model):
    """Určuje 10 nejpravděpodobnějších tříd obrázku

    Na základě modelu funkce určí 10 nejpravděpodobnějších tříd, do kterých
    obrázek spadá. Následně zprocesuje výsledek do podoby user-friendly třídy
    predikovany_objekt (či přesněji do listu s objekty tohoto typu).

    Args:
        zdrojovy_obrazek (numpy.ndarray): Obrázek - tenzor, který se snažíme zařadit
        model (keras.engine.training.Model) - použitý předtrénovaný model

    Returns:
        list: List objektů typu PredikovanyObjekt
    """
    predikce = model.predict(zdrojovy_obrazek)
    deset_nejvhodnejsich = decode_predictions(predikce, top=10)[0]
    list_predpovedi = []
    for poradi_minus_jedna, element in enumerate(deset_nejvhodnejsich):
        poradi = poradi_minus_jedna + 1
        jmeno_tridy = element[1]
        pravdepodobnost = round(100*element[2], 2)
        index = np.where(predikce == np.partition(predikce.flatten(), -2)[-poradi])[1][0]
        novy_objekt = PredikovanyObjekt(poradi, jmeno_tridy, pravdepodobnost, index)
        list_predpovedi.append(novy_objekt)
    return list_predpovedi

def urceni_trid(img_path, model):
    """Fce naplní combobox a Text widget předpověďmi tříd obrázku

    Args:
        img_path (string): Název souboru - zdrojového obrázku i s cestou
        model (keras.engine.training.Model): Použitý model
    """
    obrazek = nacti_obrazek(img_path)
    seznam = ziskej_list_predikci(obrazek, model)
    tridy_pro_combobox = []
    global global_mapovani_jmeno_na_index
    text_seznam_trid.config(state=tk.NORMAL)
    text_seznam_trid.delete(1.0, tk.END)
    for trida_obrazku in seznam:
        retezec = f"{trida_obrazku.poradi}. class {trida_obrazku.objekt} with probability {trida_obrazku.pravdepodobnost}%\n"
        text_seznam_trid.insert(tk.END, retezec)
        tridy_pro_combobox.append(trida_obrazku.objekt)
        global_mapovani_jmeno_na_index[trida_obrazku.objekt] = trida_obrazku.index
    text_seznam_trid.config(state=tk.DISABLED)
    combobox_volba_tridy["values"] = tridy_pro_combobox
    tlacitko_uloz_obrazek.config(state=tk.DISABLED)

    tlacitko_vyrob_heatmapu.config(state=tk.NORMAL)

def vyrob_feature_mapu(model, jmeno_finalni_vrstvy, index, zprocesovany_obrazek):
    """Vrací poslední konvoluční vrstvu přenásobenou její důležitostí pro třídu

    Args:
        model (keras.engine.training.Model): Použitý model
        jmeno_finalni_vrstvy (string): Jméno poslední konvoluční vrstvy modelu
        index (integer): Index třídy, pro kterou feature mapu vytváříme
        zprocesovany_obrazek (numpy.ndarray): Zprocesovaný vstupní obrázek

    Returns:
        numpy.ndarray: Feature mapa zahrnující info o tom, jak je uričtý channel \
        důležitý pro danou třídu; obvyklý tvar tensoru (14, 14, 512)
    """
    trida_output = model.output[:, index]
    last_conv_layer = model.get_layer(jmeno_finalni_vrstvy)
    pocet_filtru = last_conv_layer.filters
    #vraci gradienty trida_output pro vrstvu last_conv_layer
    grads = K.gradients(trida_output, last_conv_layer.output)[0]
    #spočte se průměr gradentů přes všechny osy krom té poslední channelové,
    #tj. výsledek má 512 prvků (počet dán vrstvou modelu)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    #fce zajišťuje přístup k výše definovaným veličinám
    iterate = K.function([model.input],
                         [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([zprocesovany_obrazek])
    #conv_layer_output_value se pro každý channel
    #přenásobuje mírou důležitosti channelu pro danou třídu
    for i in range(pocet_filtru):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    return conv_layer_output_value

def vyrob_heatmapu(feature_mapa):
    """Vyrobí heatmapu - nanormovaný průměr channelů feature mapy

    Args:
        feature_mapa (numpy.ndarray): Ovbvykle tensor floatů tvaru (14, 14, 512)

    Returns
        numpy.ndarray: Tensor floatů tvaru (14,14) s prvky z intervalu <0,1>
    """
    heatmapa = np.mean(feature_mapa, axis=-1)
    heatmapa = np.maximum(heatmapa, 0)
    heatmapa /= np.max(heatmapa)
    return heatmapa

def spoj_heatmapu_a_original(jmeno_souboru, heatmapa):
    """Načte původní obrázek a slije ho dohromady s heatmapou

    Původní obrázek se znovu načítá z disku, tj. má původní tvar a nikoli (224, 224)

    Args:
        jmeno_souboru (string): Jméno souboru s obrázkem včetně cesty
        heatmapa (numpy.ndarray): Heatmapa tvaru (14,14) zachycující pro určitou \
        třídu nejdůležitější místa obrázku
    """
    puvodni_obrazek = cv2.imread(jmeno_souboru)
    heatmapa = cv2.resize(heatmapa,
                          (puvodni_obrazek.shape[1], puvodni_obrazek.shape[0]))
    heatmapa = np.uint8(255 * heatmapa)
    heatmapa = cv2.applyColorMap(heatmapa, cv2.COLORMAP_JET)
    slozeny_obrazek = heatmapa * 0.4 + puvodni_obrazek
    return slozeny_obrazek

def nacti_obrazek_do_labelframu(adresa_souboru, jmeno_labelframu):
    """Funkce načítá obrázek z disku a umísťuje ho do zvoleného labelframu

    Původně jsem pro obrázek s heatmapou v podobě numpy.arraye konvertoval
    na Image pomocí Image.fromarray. To zahrnovalo konverzi na integer, která
    hádám přetekla přes 255 a tak se v obrázcích objevovaly nepatřičné modré fleky.
    Aktuální přístup není zrovna elegantní, ale funguje.

    Args:
        adresa_souboru (string): Jméno souboru - obrázku i s cestou
        jmeno_labelframu (tkinter.ttk.Labelframe): Objekt, do kterého se obr. vloží
    """
    nacteny_obrazek = Image.open(adresa_souboru)
    nacteny_obrazek = nacteny_obrazek.resize((SIRKA_OBRAZKU, VYSKA_OBRAZKU),
                                             Image.ANTIALIAS)
    konvertovany_obrazek = ImageTk.PhotoImage(nacteny_obrazek)
    obr_v_labelframu = ttk.Label(jmeno_labelframu, image=konvertovany_obrazek)
    obr_v_labelframu.image = konvertovany_obrazek
    obr_v_labelframu.grid(column=0, row=0)

def vyrob_vysledny_obrazek(lokace_puvodniho_obrazku, model, jmeno_finalni_vrstvy,
                           jmeno_obrazku, mapovani_jmeno_na_index):
    """Fce vyrábí obr. heatmapy přeloženou přes originál a ukládá to do tempu a okna programu

    Args:
        lokace_puvodniho_obrazku (string): Jméno souboru - obrázku i s cestou
        model (keras.engine.training.Model): Použitý model
        jmeno_finalni_vrstvy (string): Jméno poslední konvoluční vrstvy modelu
        jmeno_obrazku (string): Jméno souboru - obrázku i s cestou
        mapovani_jmeno_na_index (dict): Namapování jména třídy na index
    """
    vybrana_trida = combobox_volba_tridy.get()
    index = mapovani_jmeno_na_index[vybrana_trida]

    zprocesovany_obrazek = nacti_obrazek(lokace_puvodniho_obrazku)

    feature_mapa = vyrob_feature_mapu(model,
                                      jmeno_finalni_vrstvy,
                                      index,
                                      zprocesovany_obrazek)
    heatmapa = vyrob_heatmapu(feature_mapa)
    obr_a_heatmapa = spoj_heatmapu_a_original(jmeno_obrazku, heatmapa)
    cv2.imwrite("temp\\temporary_picture.jpg", obr_a_heatmapa)
    nacti_obrazek_do_labelframu("temp\\temporary_picture.jpg",
                                labelframe_obr_s_heatmapou)
    tlacitko_uloz_obrazek.config(state=tk.NORMAL)

def uloz_vysledny_obrazek():
    """Fce bere kopii aktuálního obrázku z tempu a ukládá ji pod jménem určeným uživatelem
    """
    jmeno_ulozeneho_souboru = fld.asksaveasfilename(initialdir=".",
                                                    title="Select file",
                                                    filetypes=[("jpeg files", "*.jpg")])
    jmeno_ulozeneho_souboru += ".jpg"
    shutil.copyfile("temp\\temporary_picture.jpg", jmeno_ulozeneho_souboru)

root = tk.Tk()
root.title("Image recognition")
mainframe = ttk.Frame(root, padding=(3, 3, 12, 12))
mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

VYSKA_OBRAZKU = 500
SIRKA_OBRAZKU = 400
global_jmeno_obrazku = ""
global_mapovani_jmeno_na_index = {}
pouzity_model = VGG16(weights='imagenet')
jmeno_fin_conv_vrstvy = 'block5_conv3'

labelframe_puvodni_obrazek = ttk.Labelframe(mainframe, text='Original image',
                                            width=SIRKA_OBRAZKU, height=VYSKA_OBRAZKU)
labelframe_puvodni_obrazek.grid_propagate(False)
labelframe_puvodni_obrazek.grid(column=0, row=0, padx=10, pady=5)
labelframe_obr_s_heatmapou = ttk.Labelframe(mainframe, text='Image with heatmap',
                                            width=SIRKA_OBRAZKU, height=VYSKA_OBRAZKU)
labelframe_obr_s_heatmapou.grid_propagate(False)
labelframe_obr_s_heatmapou.grid(column=1, row=0, padx=10, pady=5)

tlacitko_nahraj_obrazek = ttk.Button(mainframe, text="Load image", command=dej_obrazek_do_okna)
tlacitko_nahraj_obrazek.grid(column=0, row=1)
tlacitko_uloz_obrazek = ttk.Button(mainframe, text="Save image",
                                   command=uloz_vysledny_obrazek, state=tk.DISABLED)
tlacitko_uloz_obrazek.grid(column=1, row=1)

tlacitko_zjisti_tridu = ttk.Button(mainframe, text="Determine the most probable class",
                                   command=lambda: urceni_trid(global_jmeno_obrazku,
                                                               pouzity_model),
                                   state=tk.DISABLED)
tlacitko_zjisti_tridu.grid(column=0, row=2, columnspan=2)

text_seznam_trid = tk.Text(mainframe, height=10, width=50, state=tk.DISABLED)
text_seznam_trid.grid(column=0, row=3, rowspan=3)

label_volba_tridy = ttk.Label(mainframe, text="Choose class")
label_volba_tridy.grid(column=1, row=3)
combobox_volba_tridy = ttk.Combobox(mainframe, state="readonly")
combobox_volba_tridy.grid(column=1, row=4)
tlacitko_vyrob_heatmapu = ttk.Button(mainframe,
                                     text="Show heatmap for given class",
                                     command=lambda: vyrob_vysledny_obrazek(global_jmeno_obrazku,
                                                                            pouzity_model,
                                                                            jmeno_fin_conv_vrstvy,
                                                                            global_jmeno_obrazku,
                                                                            global_mapovani_jmeno_na_index),
                                     state=tk.DISABLED)
tlacitko_vyrob_heatmapu.grid(column=1, row=5)

root.mainloop()
