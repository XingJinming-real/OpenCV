import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def SIFT():
    """harris can't deal with an img when it is scaled, so a scale invarient
    algorithm is proposed called SIFT"""
