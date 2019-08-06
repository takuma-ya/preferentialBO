#!/bin/bash

METHOD="pathwise" 
export METHOD

python cgi-bin/init.py

python -m http.server --cgi
