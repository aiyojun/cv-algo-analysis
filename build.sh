#!/bin/sh

echo "-- install python dependency ..."
pip install -r requirements.txt

echo ""
echo ""

echo "-- install javascript dependency ..."
wget https://cdn.staticfile.org/echarts/4.3.0/echarts.min.js -O ./www
wget https://cdn.staticfile.org/jquery/1.10.2/jquery.min.js -O ./www

echo ""
echo ""

echo "-- prepare evironment for program ..."
mkdir images
mkdir output_images

ln -s output_images www/out
ln -s images www/pic 

echo ""
echo ""

echo "-- OK! Now, you can launch program for CV algorithm analysis!"
echo "-- Please run `python server.py`"
