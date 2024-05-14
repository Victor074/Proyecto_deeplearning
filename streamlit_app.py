import streamlit as st
import pandas as pd
import numpy as np
from skimage import io
import torch
from torchvision import transforms
import torchvision.models as models
import numpy as np
from PIL import Image
#import io
import torch.nn as nn

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
        "claude-monet",
        "camille-pissarro",
        "childe-hassam",
        "boris-kustodiev",
        "edgar-degas",
        "eugene-boudin",
        "gustave-dore",
        "ilya-repin",
        "ivan-aivazovsky",
        "ivan-shishkin",
        "john-singer-sargent",
        "marc-chagall",
        "martiros-saryan",
        "pablo-picasso",
        "nicholas-roerich",
        "paul-cezanne",
        "pierre-auguste-renoir",
        "pyotr-konchalovsky",
        "raphael-kirchner",
        "rembrandt",
        "salvador-dali",
        "hieronymus-bosch",
        "vincent-van-gogh",
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
        "claude-monet",
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


class CustomAlexNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomAlexNet, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)
        # Freeze the features layers
        for param in self.alexnet.features.parameters():
            param.requires_grad = False
        # Replace the classifier layer
        num_features = self.alexnet.classifier[6].in_features
        self.alexnet.classifier[6] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.alexnet(x)

class CustomResNet18(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.resnet18(x)
        return x

class VGGClassifier(nn.Module):
    def __init__(self, num_classes):
        super(VGGClassifier, self).__init__()
        self.vgg = models.vgg16(pretrained=True)

        for param in self.vgg.parameters():
            param.requires_grad = False

        num_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.vgg(x)

num_artist_classes = 129
num_genre_classes = 10
num_style_classes = 27

model_artist = CustomAlexNet(num_artist_classes)
model_genre = CustomResNet18(num_genre_classes)
model_style = VGGClassifier(num_style_classes)
model_artist.load_state_dict(torch.load('best_model_artist.pth'))
model_genre.load_state_dict(torch.load('best_model_genre.pth'))
model_style.load_state_dict(torch.load('best_model_style.pth'))

def predict_label(image, model):
    # Define the transformation to be applied to the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image
        transforms.ToTensor(),           # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])

    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add a batch dimension

    # Pass the preprocessed image through the model
    with torch.no_grad():
        output = model(image)

    # Convert the model output to class labels
    _, predicted = torch.max(output, 1)

    return predicted.item()

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
    st.subheader(f":blue[Dataset]")
with st.container():
    st.subheader(f":blue[Codigo]")
    st.subheader(f":blue[Arquitectura para {histo_selected}]")
    st.image(plots[histo_selected][1], width=800)


with st.container():
    st.subheader(f":blue[Demo]")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    print(uploaded_file)
    print(type(uploaded_file))
    if uploaded_file is not None:
    # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict label for the uploaded image
        if st.button("Predict"):
            predicted_label_artist = predict_label(image, model_artist)
            predicted_label_style = predict_label(image, model_style)
            predicted_label_genre = predict_label(image, model_genre)
            st.write("Predicted artist label:", all_labels['artist'][predicted_label_artist])
            st.write("Predicted style label:", all_labels['style'][predicted_label_style])
            st.write("Predicted genre label:", all_labels['genre'][predicted_label_genre])
with st.container():
    st.subheader(f":blue[Results]")
    st.write("Style label (129 labels): 54.12")
    st.write("Genre label (129 labels): 65.31")
    st.write("Artist label (129 labels): 35.83")
    