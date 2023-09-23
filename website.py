# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:40:18 2023

@author: cakir
"""

import torch
from models.generator import Generator
from models.srganGenerator import *
import torchvision
import numpy as np
from flask import Flask, render_template
from PIL import Image

netG = torch.load("weights/generator950.pt")
srganGen = torch.load("weights/SRGANgenerator64x64.pt")

def generateImg(dcganG, srganG):
    fixed_noise = torch.randn(4, 100, 1, 1, device="cuda")
    pred = dcganG(fixed_noise)
    resizedPred = srganG(pred)
    resizedPred = torch.tanh(resizedPred)
    resizedPred = resizedPred.squeeze(0)
    resizedPred = np.transpose(torchvision.utils.make_grid(resizedPred).cpu().detach().numpy(), (1, 2, 0))
    resizedPred = ((resizedPred + 1) * 127.5).astype(np.uint8)
    brightness_factor = 0.9
    darkened_image = np.clip(resizedPred * brightness_factor, 0, 255).astype(np.uint8)
    img = Image.fromarray(darkened_image)
    img.save("static/images/animeGirl.jpg")

app = Flask(__name__)

@app.route('/')
def home():
    generateImg(netG, srganGen)
    return render_template('anime.html')

if __name__ == "__main__":
    app.run()
 