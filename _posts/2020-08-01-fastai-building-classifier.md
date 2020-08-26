---
title: "Classifying Building Architecture Style with fast.ai"
date: 2020-08-01
tags: [deep learning, fast.ai, deployment]
excerpt: "Deep Learning, fast.ai Deployment"
mathjax: "true"
---

This is my first project using [fast.ai](https://docs.fast.ai/) which I started learning last month by watching the great free course from Jeremy Howard and Sylvain Gugger.  I highly recommend the coures (and the new book) for anyone interested in deep learning.

For my project I decided to use fast.ai to build an image classification model that could predict images of buildings into 4 architectural styles:
- Classical
- Gothic
- Modern
- Victorian

My code can be found in this [notebook](https://github.com/mtodisco10/fastaiProjects/blob/master/architecture_classifier.ipynb) but the general steps I took were:
1. Create a dataset of images from Google Image search
2. Download and format the images into labeled folders
3. Train the model and understand the performance
4. Deploy the model using [Render](https://render.com/)

The end result can be found [HERE](https://classifying-building-architecture.onrender.com/).  Upload an image of a building and see what the model classifies it as.

