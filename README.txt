The modeling of many Hamiltonian systems, such as pendulums in physics, predator-prey dynamics in biology, particle motion in solid-state physics, and motion of stars in astronomy, is of scientific interest for the descriptive, predictive, and prescriptive analysis of dynamical systems. Neural networks are universal function approximations that can learn the dynamics of continuous dynamical systems, including Hamiltonian systems. Furthermore, the neural network of a dynamical system can be informed of its invariances to improve its modeling. One invariance that is common in Hamiltonian systems is periodicity. We propose several methods to embed periodicity in neural networks. We compare their performance on interpolating and extrapolating the dynamical system. Our methods can reduce the error in extrapolating the Hamiltonian from $10$ to less than $0.1$ and the error in extrapolating the vector field from $100$ to less than $1$. 

The main contributions of this work are:
\begin{itemize} 
    \item  We propose periodic HNNs that embed periodicity using three modes of biases: observational bias, learning bias, and inductive bias. 
    \item An observational bias is embedded by training the HNN on newly generated data that embody periodicity. A learning bias is embedded through the loss function of the HNN. An inductive bias is embedded by the activation function in the HNN. Each proposed periodic HNN may embed one or more biases. 
    \item The proposed periodic HNNs show high performance in four tasks. The four tasks are the interpolation and extrapolation of the Hamiltonians and vector fields of periodic Hamiltonian systems, with or without the knowledge of their periods. The code for the HNNs is available at \url{github.com/zykhoo/periodicNNs}.
\end{itemize}


Four experiments are performed corresponding to the four tasks mentioned above. The four experiments compare the performance of the baseline and proposed models on the interpolation and extrapolation of the Hamiltonians and vector fields of periodic Hamiltonian systems, with or without the knowledge of their periods, respectively. The data for all experiments was simulated and can be found in the folder ~/periodicHNNknown/Experiments/Baseline/data in this repository.

Thank you for your interest in our work!
