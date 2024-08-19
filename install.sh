#!/bin/bash

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null
then
    echo "Python 3 could not be found. Please install Python 3 and try again."
    exit
fi


echo "Creating virtual environment..."
python3 -m venv venv


source venv/bin/activate


echo "Updating pip..."
pip install --upgrade pip


echo "Installing Tkinter..."
sudo apt-get update
sudo apt-get install python3-tk -y


echo "Installing required Python packages..."
pip install Pillow opencv-python ultralytics numpy


echo "Running yoloGui.py..."
python yoloGui.py


deactivate
