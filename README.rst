######
phygnn
######

phygnn stands for **Physics-Guided Neural Networks**.

This implementation of Physics-Guided Neural Networks augments a traditional 
neural network loss function with a generic loss term that can be used to 
guide the neural network to learn physical or theoretical constraints.
phygnn enables scientific software developers and data scientists to easily 
integrate machine learning models into physics and engineering applications.

For details on the phygnn class framework see `the phygnn module documentation here. <https://nrel.github.io/phygnn/phygnn/phygnn.phygnn.html>`_


Installation
============


Simple Install
--------------

1. Use conda (anaconda or miniconda with python 3.7 or 3.8) to create a phygnn environment: ``conda create --name phygnn python=3.8``
2. Activate your new conda env: ``conda activate phygnn``
3. Install with pip or conda:

    i. ``pip install NREL-phygnn``
    ii. ``conda install -c nrel nrel-phygnn``


Developer Install
-----------------

1. Use conda (anaconda or miniconda with python 3.7) to create a phygnn environment: ``conda create --name phygnn python=3.8``
2. Activate your new conda env: ``conda activate phygnn``
3. Clone the phygnn repository: ``git clone https://github.com/NREL/phygnn.git`` or ``git clone git@github.com:NREL/phygnn.git``
4. Navigate to the cloned repo and checkout your desired branch: ``git checkout master`` or ``git checkout <branch>``
5. Navigate to the phygnn directory that contains setup.py and run: ``pip install -e .`` (developer install) or ``pip install .`` (static install).
6. Test your installation:

    i. Start ipython and test the following import: ``from phygnn import PhysicsGuidedNeuralNetwork``
    ii. Navigate to the ``tests/`` directory and run the command: ``pytest``
