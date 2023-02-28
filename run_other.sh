# Tests of MATLAB code (LAND, CAL) and S^2 algorithm

# first, convert matlab results to python results directory
python convert_mat_to_python.py --resultsdir results_s2 --save_acc 1


python test_al_s2.py --dataset box --metric raw --K 8 --config config_other.yaml --resultsdir results_other 
python accuracy_al_gl.py --dataset box --metric raw --config config_other.yaml --resultsdir results_other
python compile_summary.py --dataset box --resultsdir results_other

python test_al_s2.py --dataset blobs --metric raw --K 16 --config config_other.yaml --resultsdir results_other 
python accuracy_al_gl.py --dataset blobs --metric raw --config config_other.yaml --resultsdir results_other
python compile_summary.py --dataset blobs --resultsdir results_other


# did not run S2 on isolet, so just run accuracy on others. Need to change config file to also have "laplace" accuracy model
python accuracy_al_gl_isolet.py --dataset isolet --metric raw --config config_other.yaml --resultsdir results_other
python compile_summary.py --dataset isolet --resultsdir results_other


