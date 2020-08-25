######
phygnn
######

phygnn stands for **physics-guided neural networks**.

This implementation of physics-guided neural networks augments a traditional
neural network loss function with a generic loss term that can be used to
guide the neural network to learn physical or theoretical constraints.
phygnn enables scientific software developers and data scientists to easily
integrate machine learning models into physics and engineering applications.
This framework should help alleviate some challenges that are often encountered
when applying purely data-driven machine learning models to scientific
applications, such as when machine learning models produce physically
inconsistent results or have trouble generalizing to out-of-sample scenarios.

For details on the phygnn class framework see `the phygnn module documentation here. <https://nrel.github.io/phygnn/phygnn/phygnn.phygnn.html>`_

At the National Renewable Energy Lab (NREL), we are using the phygnn framework
to supplement traditional satellite-based cloud property prediction models. We
use phygnn to predict cloud optical properties when the traditional mechanistic
models fail and use a full tensor-based radiative transfer model as the
physical loss function to transform the predicted cloud properties into
phygnn-predicted irradiance data. We then calculate a loss value comparing the
phygnn-predicted irradiance to high quality ground measurements. We have seen
excellent improvements in the predicted irradiance data in rigorous
out-of-sample-validation experiments.

Engineers and researchers can use the phygnn framework to:

    * Enforce physically-consistent predictions from a deep neural network (see lake temperature reference below)
    * Use the physics loss function to extend training data, e.g. train against "known" outputs but also train using the downstream application of the predicted variables
    * Use the physics loss function to adjust theoretical models based on empirical observation using respective loss weights

Here are additional examples of similar architectures from the literature which
helped inspire this work:

    * Jared Willard, Xiaowei Jia, Shaoming Xu, Michael Steinbach, and Vipin Kumar, “Integrating Physics-Based Modeling with Machine Learning: A Survey.” ArXiv abs/2003.04919 (2020).
    * Forssell, U. and P. Lindskog. “Combining Semi-Physical and Neural Network Modeling: An Example ofIts Usefulness.” IFAC Proceedings Volumes 30 (1997): 767-770.
    * Xinyue Hu, Haoji Hu, Saurabh Verma, and Zhi-Li Zhang, “Physics-Guided Deep Neural Networks for PowerFlow Analysis”, arXiv:2002.00097v1 (2020).
    * Anuj Karpatne, William Watkins, Jordan Read, and Vipin Kumar, "Physics-guided Neural Networks (PGNN): An Application in Lake Temperature Modeling". arXiv:1710.11431v2 (2018).
    * Anuj Karpatne, Gowtham Atluri, James H Faghmous, Michael Steinbach, Arindam Banerjee, Auroop Ganguly, Shashi Shekhar, Nagiza Samatova, and Vipin Kumar. 2017. Theory-guided data science: A new paradigm for scientific discovery from data. IEEE Transactions on knowledge and data engineering 29, 10 (2017), 2318–2331.


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
