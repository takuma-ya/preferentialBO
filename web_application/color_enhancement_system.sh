#!/bin/bash

METHOD="linewise" 
export METHOD

python cgi-bin/init.py

python -m http.server --cgi
