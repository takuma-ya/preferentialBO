## Environment

- python3.7
- colormath3.0.0
- numpy1.15.1
- scipy1.1.0
- DIRECT1.0.0
- PIL5.2.0
- sklearn0.19.2
  

## Usage

1. ./color_enhancement_system.sh 
2. launch a browser
    http://localhost:{port number:8000}/slider.html
3. calculate color difference from selected_param.txt
    python color_difference_cie2000.py


## Log Files

- history.txt: used in program
- negative_representives.npy:	used in program
- path.npy: used in program
- residual.txt: residuals from optimal parameter 
- residual_cie2000.txt: color_difference from optimal parameter
- selected_param.txt: selected parameters from a user


## Program Files
- color_difference_cie2000.py: calculate color difference
- init.py: initialization 
- step.py: called at each round
- color_enhance.py: library file for run color enhancement from parameters	
- interactive_bo.py: library file for interactive BO


## Image Dir

- original: there is an original picture
- reference: there is an reference picture
- slider: there are generated pictures corresponding to a slider
