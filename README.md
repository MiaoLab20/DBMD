# Deep Boosted Molecular Dynamics: Accelerating molecular simulations with Gaussian boost potentials generated using probabilistic Bayesian deep neural network
We have developed a new Deep Boosted Molecular Dynamics (DBMD) method. Probabilistic Bayesian neural network models were implemented to construct boost potentials that exhibit Gaussian distribution with minimized anharmonicity, thereby allowing for accurate energetic reweighting and enhanced sampling of molecular simulations. We have successfully demonstrated DBMD on a wide range of model systems, including alanine dipeptide and the fast-folding protein and RNA structures. Based on Deep Learning neural network, DBMD provides a powerful and generally applicable approach to boosting biomolecular simulations.

An example ***test*** folder that contains the input topology and coordinate file for the folding simulation of the hairpin RNA with GCAA tetraloop is included in this repository. The following *python* modules must be installed to perform a DBMD simulation:
* OpenMM: https://openmm.org/
* TensorFlow: https://www.tensorflow.org/install
* TensorFlow Keras: https://www.tensorflow.org/api_docs/python/tf/keras
* TensorFlow Probability: https://www.tensorflow.org/probability/install
* Scikit-learn: https://scikit-learn.org/stable/install.html
* Numpy: https://numpy.org/install/
* Pandas: https://pandas.pydata.org/getting_started.html (*panda is also my favorite animal*)

* <img width="174" alt="image" src="https://user-images.githubusercontent.com/57517329/232138322-8b7856bc-6060-4133-8c4f-41af00333c55.png">

* Matplotlib: https://matplotlib.org/
* Seaborn: https://seaborn.pydata.org/installing.html

It is recommended to install these modules and run DBMD in OpenMM in an Anaconda environment: https://www.anaconda.com/

An example input file for an DBMD simulation can be found in ***simParams.py***. A run script can be found in ***runSimulation***. To run the ***test*** folder, simply install all the necessary *python* modules and run the following commands:

***sh runSimulation***

Explanations for all parameters in the example input file can be found at the reference below. It is recommended to set up and run DBMD in OpenMM on NVIDIA GPUs to achieve the best possible speeds.

# Reference
Do, H.N. and Miao, Y. (2023) Deep Boosted Molecular Dynamics (DBMD): Accelerating molecular simulations with Gaussian boost potentials generated using probabilistic Bayesian deep neural network. *bioRxiv*. https://www.biorxiv.org/content/10.1101/2023.03.25.534210v2
