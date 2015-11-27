# Facial attributes classifiers
This is the code for the following paper:
> P. Samangouei, V. M. Patel, and R. Chellappa. Attribute-based active authentication on mobile devices. In Biometrics: Theory, Applications and Systems–BTAS. 2015.

Note: it's a research code and will evolve over time. If you face any issues compiling it, or incomplete documentation, please contact me.

Although it has both the training and testing code, main.cpp is just for getting the attributes for a list of cropped and aligned faces.
> $ ./main cls_paths images_path output_path

The crop and align code is in dlibalign.cpp file.

The makefile works on my RHEL7 machine, you may need to change it to get it working for yourself.


#Prereq:
* OpenCV
* DLib



#TODO
Document functions
Make test cases
