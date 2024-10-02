for sys in "pendulum" "System3" "System5" "System11" "System12" "doublepend" # # "pendulum" "System3" # "pendulum" "trigo" "arctan" "System2" "System3" "System4" "System5" "System6" "System7" "System8" "System9" "System10" "System11" "System12" "System13" "System14"
do
  mkdir $sys
  sed -i "37s/.*/s = '$sys'/" Experiment.py
  #sed -i "37s/.*/s = '$sys'/" ExperimentLearning.py
  CUBLAS_WORKSPACE_CONFIG=:16:8 python Experiment.py 
  #CUBLAS_WORKSPACE_CONFIG=:16:8 python ExperimentLearning.py 
done


#sed -i '36s/.*/s = "trigo"/' Experiment.py
#CUBLAS_WORKSPACE_CONFIG=:16:8 python Experiment.py 

#sed -i '36s/.*/s = "anisotropicoscillator2D"/' Experiment.py
#CUBLAS_WORKSPACE_CONFIG=:16:8 python Experiment.py 

#sed -i '36s/.*/s = "todalattice"/' Experiment.py
#CUBLAS_WORKSPACE_CONFIG=:16:8 python Experiment.py 
