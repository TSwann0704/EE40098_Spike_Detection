INSTRUCTIONS FOR RUNNING THE CODE. DEVELOPED USING VISUAL STUDIO CODE

1. Install the following libraries in local python environment:
    - numpy 
    - scipy
    - tensorflow
    - tensorflow.keras
    - sklearn
    - random

2.  Import signal data files as D1.mat, D2.mat, D3.mat, D4.mat, D5.mat, D6.mat 

3.  Install the CourseworkC_GENERATE_MODEL.py and CourseworkC_APPLY_MODEL.py files into the 
    same directory. Ensure the directory has read/write priviliges. Should have the following structure:

    /folder
        D1.mat
        D2.mat
        D3.mat
        D4.mat
        D5.mat
        D6.mat
        CourseworkC_GENERATE_MODEL.py
        CourseworkC_APPLY_MODEL.py

4. Run CourseworkC_GENERATE_MODEL.py, check CNN_Model.h5 file is saved.

5. Run CourseworkC_APPLY_MODEL.py, check .mat file is saved. 

6. Repeat step 5 for each dataset, update the following variables in CourseworkC_APPLY_MODEL.py:
    - Line 229: Replace .mat file with title of the dataset.
    - Line 50-59: Replace filter constants with corresponding to target dataset.
    - Line 95-104: Replace amplitude cut off constant as labelled for each dataset.