This is a repository for a piece of software written in python, designed to generate terrain heightmaps using realistic simulation of river networks.

Run the application with 

$ python main.py

It does not currently support any tests. I will perform the testing manually. Just run the application to check for syntax errors.

The source code for the application is in ./terrain_generator. It is organized into config, core, gui, and io files, located in their respective
directiories (/config, /core, /gui, io, /visualization). Each part contains the code for that part of the application. The presets directory holds presets for the terrain generator. The core directory has the main code for generating noise, generating rivers, generating the terrain, and utils. The gui folder contains code for the PyQT5 graphical UI. The io directory has stuff related to importing and exporting PNG images. The visualization directory contains code for rendering the 3d preview of the terrain in the viewport of the GUI.