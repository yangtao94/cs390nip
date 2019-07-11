<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://imgur.com/wxW7qoK.png" alt="Project logo"></a>
</p>

<h3 align="center">Style Transfer</h3>

<div align="center">

  [![Status](https://img.shields.io/badge/status-active-success.svg)]() 
  [![GitHub Issues](https://img.shields.io/github/issues/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/pulls)
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> Transfer the style of other images onto your own.
    <br> 
</p>

## 📝 Table of Contents
- [About](#about)
- [Getting Started](#getting_started)
- [Deployment](#deployment)
- [Usage](#usage)
- [Built Using](#built_using)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>
What is Style Transfer? It is the technique of recomposing images in the style of other images. 

A simple disco ball is restyled to the style represented by Gustav Klimt's "The Kiss"
![Transfer!](http://genekogan.com/images/style-transfer/mrdiv-klimt.gif)

In our model, however, we will only be doing Style Transfer for images.

## 🏁 Getting Started <a name = "getting_started"></a>
Clone the repository! the main jupyter notebook file is Lab2.ipynb. 


### Prerequisites
Anaconda; Python, Jupyter Notebook

Install <a href= "https://www.anaconda.com/distribution/">Anaconda</a> and the rest are all installed together

```
jupyter notebook
```

in the Anaconda terminal to open up the jupyter notebook

If you can't seem to open up jupyter notebook on github, go to https://nbviewer.jupyter.org/ and paste the github link in there.

Install numpy in the terminal

```
conda install numpy
```


Install pandas

```
conda install pandas
```

Install matplotlib

```
conda install -c conda-forge matplotlib
```

Install tensorflow

```
conda create -n tensorflow_env tensorflow
 	
  conda activate tensorflow_env
```
### Installing

Run 

```
python Lab2.py
```

to start the transfer process!




## 🎈 Usage <a name="usage"></a>

<img src = "https://imgur.com/bGeih11.png">

Change the CONTENT_IMG_PATH value to your own desired image to be transformed.

Change the STYLE_IMG_PATH value to your own desired style image

The value of STYLE_WEIGHT and CONTENT_WEIGHT corresponds to the content loss function and the style loss function.

The content loss function and the style loss function. The content loss function ensures that the activations of the higher layers are similar between the content image and the generated image. The style loss function makes sure that the correlation of activations in all the layers are similar between the style image and the generated image. 

<a href = "https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-neural-style-transfer-ef88e46697ee"> More information here </a>

In other words, if you want the priortize style over content in the final image, increase the value of STYLE_WEIGHT

If you want to make the transfer process longer (especially if your computer is good enough), then increase the value of TRANSFER_ROUNDS! Note : For every transfer round a picture will be saved.

Heres a few images I made : Also in the repository

Style Image : Mona Lisa

<img src = "https://imgur.com/PIQl3Uo.png">

Content Image : Ben Simmons, a NBA player that many believe looks like Mona Lisa herself

<img src = "https://imgur.com/kIYIuHb.png">

Transferred Image :

<img src = "https://imgur.com/O8ITzlW.png">

Style Image : Van Gogh's Starry Night

<img src = "https://images-na.ssl-images-amazon.com/images/I/61ySbUOxYRL.jpg">

Content Image : Kobe Bryant shooting over 3 defenders like he always do

<img src = "https://imgur.com/VqU5YDz.png">

Transferred Image:

<img src = "https://imgur.com/K2VhZxz.png">

Make your own!!



## ⛏️ Built Using <a name = "built_using"></a>
- [Python]

## 🎉 Acknowledgements <a name = "acknowledgement"></a>
- Shout out to <a href = "https://keras.io/examples/neural_style_transfer/"> Keras </a> for making life so simple
