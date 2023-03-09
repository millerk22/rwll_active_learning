# python test_al_gl.py --dataset mnist-mod3 --K 20 --config config.yaml --resultsdir results 
# python accuracy_al_gl.py --dataset mnist-mod3 --config config.yaml --resultsdir results
# python compile_summary.py --dataset mnist-mod3 --resultsdir results

# python test_al_gl.py --dataset fashionmnist-mod3 --K 20 --config config.yaml --resultsdir results 
# python accuracy_al_gl.py --dataset fashionmnist-mod3 --config config.yaml --resultsdir results
# python compile_summary.py --dataset fashionmnist-mod3 --resultsdir results

# python test_al_gl.py --dataset emnist-mod5 --iters 400 --K 94 --config config.yaml --resultsdir results 
# python accuracy_al_gl.py --dataset emnist-mod5 --iters 400 --config config.yaml --resultsdir results
# python compile_summary.py --dataset emnist-mod5 --iters 400 --resultsdir results


# python test_al_gl.py --dataset box --metric raw --K 8 --config config.yaml --resultsdir results 
# python accuracy_al_gl.py --dataset box --metric raw --config config.yaml --resultsdir results
# python compile_summary.py --dataset box --resultsdir results

# python test_al_gl.py --dataset blobs --metric raw --K 16 --config config.yaml --resultsdir results 
# python accuracy_al_gl.py --dataset blobs --metric raw --config config.yaml --resultsdir results
# python compile_summary.py --dataset blobs --resultsdir results


# python test_al_gl.py --dataset mnistimb-mod3 --K 30 --config config.yaml --resultsdir results 
# python accuracy_al_gl.py --dataset mnistimb-mod3 --config config.yaml --resultsdir results
# python compile_summary.py --dataset mnistimb-mod3 --resultsdir results

# python test_al_gl.py --dataset fashionmnistimb-mod3 --K 30 --config config.yaml --resultsdir results 
# python accuracy_al_gl.py --dataset fashionmnistimb-mod3 --config config.yaml --resultsdir results
# python compile_summary.py --dataset fashionmnistimb-mod3 --resultsdir results

# python test_al_gl.py --dataset emnistvcd --iters 400 --K 120 --config config.yaml --resultsdir results 
# python accuracy_al_gl.py --dataset emnistvcd --iters 400 --config config.yaml --resultsdir results
# python compile_summary.py --dataset emnistvcd --iters 400 --resultsdir results




# Isolet test -- we include "laplace" learning model for accuracy comparison to original
python test_al_gl_isolet.py --dataset isolet --metric raw --K 50 --config config_isolet.yaml --resultsdir results 
python accuracy_al_gl_isolet.py --dataset isolet --metric raw --config config_isolet.yaml --resultsdir results
python compile_summary.py --dataset isolet --resultsdir results


## VOpt and SigmaOpt tests with "full" computations on random subsets of unlabeled data 
# python test_al_gl_voptfull.py --dataset mnist-mod3 --config config.yaml --resultsdir results_
# python accuracy_al_gl.py --dataset mnist-mod3 --config config.yaml --resultsdir results
# python compile_summary.py --dataset mnist-mod3 --resultsdir results

# python test_al_gl_voptfull.py --dataset fashionmnist-mod3 --config config.yaml --resultsdir results 
# python accuracy_al_gl.py --dataset fashionmnist-mod3 --config config.yaml --resultsdir results
# python compile_summary.py --dataset fashionmnist-mod3 --resultsdir results

#python test_al_gl_voptfull.py --dataset mnistimb-mod3 --config config.yaml --resultsdir results 
#python accuracy_al_gl.py --dataset mnistimb-mod3 --config config.yaml --resultsdir results
#python compile_summary.py --dataset mnistimb-mod3 --resultsdir results

#python test_al_gl_voptfull.py --dataset fashionmnistimb-mod3 --config config.yaml --resultsdir results 
#python accuracy_al_gl.py --dataset fashionmnistimb-mod3 --config config.yaml --resultsdir results
#python compile_summary.py --dataset fashionmnistimb-mod3 --resultsdir results



