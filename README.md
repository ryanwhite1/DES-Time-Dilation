# Cosmological Time Dilation with DES 
## Motivation
Our best cosmological models predict that as a consequence of the expansion of the universe, a clock will be observed to tick slower when viewed from cosmologically far away. In this project (paper forthcoming), we use the light curves of the ~1600 Type Ia supernovae from the Dark Energy Survey as our `standard clocks'. In comparing the duration of the light curve evolution with redshift, we find a significant time dilation signature which is consistent with previous work on the topic and helps validate our models of the Universe. 

![timedilation](/Images/AllAveWidths-vs-1+z-first1504.png)

This repository is intended only as a public record of _all_ of the code used in the project, and for the generation and storage of the figures in the paper (as well as for those that didn't make the cut!). The data used in the code is not in this repository (see Sanchez et al 2024). 
## Code
 - The `Methods.py` file contains some useful functions that are used throughout other plotting/data operations.
 - The `StretchMethod.py` file is where the widths are calculated and saved (pickled).
 - The `Plot_Generation.py` is where all of the figures are generated and saved (using the pickled/supplied data). 
