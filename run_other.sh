# Tests of MATLAB code (LAND, CAL) and S^2 algorithm

python test_al_s2.py --dataset box --metric raw --K 8 --config config_other.yaml --resultsdir results_other 
python convert_mat_to_python.py --resultsdir results_other --save_acc 1 --only box  # convert matlab results to python results directory
python accuracy_al_gl.py --dataset box --metric raw --config config_other.yaml --resultsdir results_other
python compile_summary.py --dataset box --resultsdir results_other

#python test_al_s2.py --dataset blobs --metric raw --K 16 --config config_other.yaml --resultsdir results_other
#python convert_mat_to_python.py --resultsdir results_other --save_acc 1 --only blobs  # convert matlab results to python results directory
#python accuracy_al_gl.py --dataset blobs --metric raw --config config_other.yaml --resultsdir results_other
#python compile_summary.py --dataset blobs --resultsdir results_other


# did not run S2 on isolet, so just run accuracy on others. Need to change config file to also have "laplace" accuracy model
#python convert_mat_to_python.py --resultsdir results_other --save_acc 1 --only isolet  # convert matlab results to python results directory
#python accuracy_al_gl_isolet.py --dataset isolet --metric raw --config config_other.yaml --resultsdir results_other
#python compile_summary.py --dataset isolet --resultsdir results_other


