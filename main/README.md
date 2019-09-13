# Interactive BO for setwise, linewise, and pathwise comparison.
## how to write history file 

* This system gets observed information from a text file and returns the next candidates.
* The history file should be deleted when you start another optimization.

* A history file in which observed information is saved need to be written like follows. Any number of inputs can be written as selected inputs and unselected inputs.

(selected input 1) (selected input 2)>(not selected input 1) (not selected input 2)

(selected input 1) (selected input 2) (selected input 3)>(not selected input 1)

(selected input 1)>(not selected input 1) (not selected input 2)>(not selected input 3)

                        .
  
                        . 
 
                        .  

   
* ex. pairwise comparison case, 

(selected input)>(not selected input)

(selected input)>(not selected input)

(selected input)>(not selected input)

                        .  

                        .  

                        .  


* Environment

python '3.6.5'

numpy '1.12.1'

scipy '0.19.1'

sklearn '0.19.0'

DIRECT '1.0'

* Reference 
linewise: https://koyama.xyz/project/sequential_line_search/download/preprint.pdf)

setwise, pathwise: (not yet)
