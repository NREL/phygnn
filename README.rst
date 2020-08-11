******
PHYGNN
******

An open source implementation of Physics-Guided Neural Networks (PHYGNN). 
This implementation of PHYGNN augments a traditional neural network loss 
function with a generic loss term that can be used to guide the neural 
network to learn physical or theoretical constraints.

.. inclusion-intro

Installation
============

1. Use conda (anaconda or miniconda with python 3.7) to create a phygnn environment: ``conda create --name phygnn python=3.7``
2. Activate your new conda env: ``conda activate phygnn``
3. Navigate to the phygnn directory that contains setup.py and run: ``pip install -e .`` (developer install) or ``pip install .`` (static install).
4. Test your installation:

	i. Start ipython and test the following import: ``from phygnn import PhysicsGuidedNeuralNetwork``
	ii. Navigate to the tests/ directory and run the command: ``pytest``
