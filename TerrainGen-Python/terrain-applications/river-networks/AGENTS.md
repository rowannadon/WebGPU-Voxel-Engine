This is a repository for a piece of software written in python, designed to generate terrain heightmaps using realistic simulation of river networks.

Test the application with python3 py_compile just to test for syntax errors.

The source code for the application is in ./terrain_generator. It is organized into config, core, gui, and io files, located in their respective
directiories (/config, /core, /gui, io, /visualization). Each part contains the code for that part of the application. The presets directory holds presets for the terrain generator. The core directory has the main code for generating noise, generating rivers, generating the terrain, and utils. The gui folder contains code for the PyQT5 graphical UI. The io directory has stuff related to importing and exporting PNG images. The visualization directory contains code for rendering the 3d preview of the terrain in the viewport of the GUI.

A separate part of the code is focused on generated terrain heuristic data, such as slope, normal, curvature, etc. and simulating climate and biome parameters. It is located in ./terrain_heuristucs. It implements a GUI and CLI interface of its own currently, while also being integrated into the main GUI for the terrain generation program. 