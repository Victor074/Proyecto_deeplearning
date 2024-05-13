import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io

sns.set_theme(style = "darkgrid" )
#Lectura de la Imagen

Logo = io.imread("./Pictures/monet.jpg")
st.set_page_config(page_title="Clasification de pinturas", page_icon=":tada:", layout="wide")
plot_art = io.imread("./Pictures/artistas.png")
plot_gen = io.imread("./Pictures/genero.png")
plot_sty = io.imread("./Pictures/estilo.png")
resnet = io.imread("./Pictures/ResNet18.jpg")
vgg = io.imread("./Pictures/VGGNet.jpg")
alexnet = io.imread("./Pictures/alexnet.jpg")
plots = {"Artistas":(plot_art, alexnet),"Generos":(plot_gen, resnet),"Estilos":(plot_sty, vgg)}
all_labels = {
    "artist":[
        "boris-kustodiev",
        "camille-pissarro",
        "childe-hassam",
        "claude-monet",
        "edgar-degas",
        "eugene-boudin",
        "gustave-dore",
        "ilya-repin",
        "ivan-aivazovsky",
        "ivan-shishkin",
        "john-singer-sargent",
        "marc-chagall",
        "martiros-saryan",
        "nicholas-roerich",
        "pablo-picasso",
        "paul-cezanne",
        "pierre-auguste-renoir",
        "pyotr-konchalovsky",
        "raphael-kirchner",
        "rembrandt",
        "salvador-dali",
        "vincent-van-gogh",
        "hieronymus-bosch",
        "leonardo-da-vinci",
        "albrecht-durer",
        "edouard-cortes",
        "sam-francis",
        "juan-gris",
        "lucas-cranach-the-elder",
        "paul-gauguin",
        "konstantin-makovsky",
        "egon-schiele",
        "thomas-eakins",
        "gustave-moreau",
        "francisco-goya",
        "edvard-munch",
        "henri-matisse",
        "fra-angelico",
        "maxime-maufra",
        "jan-matejko",
        "mstislav-dobuzhinsky",
        "alfred-sisley",
        "mary-cassatt",
        "gustave-loiseau",
        "fernando-botero",
        "zinaida-serebriakova",
        "georges-seurat",
        "isaac-levitan",
        "joaquadn-sorolla",
        "jacek-malczewski",
        "berthe-morisot",
        "andy-warhol",
        "arkhip-kuindzhi",
        "niko-pirosmani",
        "james-tissot",
        "vasily-polenov",
        "valentin-serov",
        "pietro-perugino",
        "pierre-bonnard",
        "ferdinand-hodler",
        "bartolome-esteban-murillo",
        "giovanni-boldini",
        "henri-martin",
        "gustav-klimt",
        "vasily-perov",
        "odilon-redon",
        "tintoretto",
        "gene-davis",
        "raphael",
        "john-henry-twachtman",
        "henri-de-toulouse-lautrec",
        "antoine-blanchard",
        "david-burliuk",
        "camille-corot",
        "konstantin-korovin",
        "ivan-bilibin",
        "titian",
        "maurice-prendergast",
        "edouard-manet",
        "peter-paul-rubens",
        "aubrey-beardsley",
        "paolo-veronese",
        "joshua-reynolds",
        "kuzma-petrov-vodkin",
        "gustave-caillebotte",
        "lucian-freud",
        "michelangelo",
        "dante-gabriel-rossetti",
        "felix-vallotton",
        "nikolay-bogdanov-belsky",
        "georges-braque",
        "vasily-surikov",
        "fernand-leger",
        "konstantin-somov",
        "katsushika-hokusai",
        "sir-lawrence-alma-tadema",
        "vasily-vereshchagin",
        "ernst-ludwig-kirchner",
        "mikhail-vrubel",
        "orest-kiprensky",
        "william-merritt-chase",
        "aleksey-savrasov",
        "hans-memling",
        "amedeo-modigliani",
        "ivan-kramskoy",
        "utagawa-kuniyoshi",
        "gustave-courbet",
        "william-turner",
        "theo-van-rysselberghe",
        "joseph-wright",
        "edward-burne-jones",
        "koloman-moser",
        "viktor-vasnetsov",
        "anthony-van-dyck",
        "raoul-dufy",
        "frans-hals",
        "hans-holbein-the-younger",
        "ilya-mashkov",
        "henri-fantin-latour",
        "m.c.-escher",
        "el-greco",
        "mikalojus-ciurlionis",
        "james-mcneill-whistler",
        "karl-bryullov",
        "jacob-jordaens",
        "thomas-gainsborough",
        "eugene-delacroix",
        "canaletto"
      ],
    "genre":[
        "abstract_painting",
        "cityscape",
        "genre_painting",
        "illustration",
        "landscape",
        "nude_painting",
        "portrait",
        "religious_painting",
        "sketch_and_study",
        "still_life",
      ],
    "style": [
        "Abstract_Expressionism",
        "Action_painting",
        "Analytical_Cubism",
        "Art_Nouveau",
        "Baroque",
        "Color_Field_Painting",
        "Contemporary_Realism",
        "Cubism",
        "Early_Renaissance",
        "Expressionism",
        "Fauvism",
        "High_Renaissance",
        "Impressionism",
        "Mannerism_Late_Renaissance",
        "Minimalism",
        "Naive_Art_Primitivism",
        "New_Realism",
        "Northern_Renaissance",
        "Pointillism",
        "Pop_Art",
        "Post_Impressionism",
        "Realism",
        "Rococo",
        "Romanticism",
        "Symbolism",
        "Synthetic_Cubism",
        "Ukiyo_e"
      ]
  }
# Header
with st.container():
    st.image(Logo, width=800)
    st.title("Clasificador de pinturas")
    st.markdown(":blue[Este proyecto clasifica pinturas]")
    st.subheader("Introduction")
    st.write(
            """
            Propongo este proyecto clasificador de pinturas basado en artistas, géneros y estilos con el propósito de ayudar 
            a catalogar y organizar grandes colecciones de arte. En el contexto de museos y galerías, la aplicación de este 
            clasificador permitirá una eficiente categorización de las pinturas, facilitando la gestión de inventarios y 
            mejorando la accesibilidad a la información relacionada con cada obra. La capacidad de identificar automáticamente 
            artistas, géneros y estilos contribuirá a agilizar los procesos de documentación y preservación, permitiendo una 
            gestión más efectiva de las valiosas colecciones artísticas. Otro de los campos donde el proyecto es de utilidad 
            es como herramienta educacional para estudiantes interesados en explorar diferentes artistas, géneros o estilos.
            """
        )
    st.markdown("**Paintings Data**")
    st.markdown(":blue[Este dataset contiene 81444 pinturas de diferentes artistas, estas pinturas se han tomado de WikiArt.org y el dataset fue recopilado en huggingface.com]")
    #st.dataframe(data)

st.sidebar.markdown("## PARAMETERS")
vars_feature = ['Artistas','Estilos','Generos']
default_hist = vars_feature.index('Artistas')
histo_selected = st.sidebar.selectbox('Label:', vars_feature, index = default_hist)

st.write("[Github repo](https://github.com/Victor074/Proyecto_deeplearning)")
with st.container():
    st.subheader(f":blue[Distribucion de {histo_selected}]")
    st.image(plots[histo_selected][0], width=800)

with st.container():
    st.subheader(f":blue[Como arreglar el desbalance?]")
    st.write("SMOTE")

with st.container():
    st.subheader(f":blue[Arquitectura para {histo_selected}]")
    st.image(plots[histo_selected][1], width=800)
with st.container():
    st.subheader(f":blue[Demo]")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        st.write("Uploaded Image:")

        # To display the uploaded image:
        st.image(bytes_data)

        if st.button('Click Me'):
            st.write('The button was clicked!')
