This repository contains a set of python scripts which generate various terrain heuristics, simulated climate, and biome information given an input heightmap. These are then rendered to various output textures. 

1. The file terrain_heuristics.py implements a CLI program which generates the various heuristics given the input heightmap. Each generation stage is separated into its own file.
The generated heuristics are:
- Ocean/land mask
- coastline outline
- Slope in degrees
- Aspect
- Normal map
- TWI (terrain wetness index)
- TPI (Topographic position index) at various radii
- Flow accumulation
- SVF (Sun visibikuty factor)
- Climate maps:
    - temperature
    - evapotranspiration
    - precipitation
    - aridity
    - distance from coastline
- Biome maps (ID and RGB), based on classification from the calculated climate maps
- Foliage color map
- Tree density
- Groundcover density

2. The file terrain_gui.py implements a GUI frontend for running the generation and viewing the output. It supports configuration of the CLI parameters. The output of each stage can be viewed, panned and zoomed.

No need to test the scripts, I will test the 