python test_al_gl.py --dataset mnist-mod3 --K 20 --config config.yaml --resultsdir results_decay 
python accuracy_al_gl.py --dataset mnist-mod3 --config config.yaml --resultsdir results_decay
python compile_summary.py --dataset mnist-mod3 --resultsdir results_decay

python test_al_gl.py --dataset fashionmnist-mod3 --K 20 --config config.yaml --resultsdir results_decay 
python accuracy_al_gl.py --dataset fashionmnist-mod3 --config config.yaml --resultsdir results_decay
python compile_summary.py --dataset fashionmnist-mod3 --resultsdir results_decay

python test_al_gl.py --dataset emnist-mod5 --iters 400 --K 94 --config config.yaml --resultsdir results_decay 
python accuracy_al_gl.py --dataset emnist-mod5 --iters 400 --config config.yaml --resultsdir results_decay
python compile_summary.py --dataset emnist-mod5 --iters 400 --resultsdir results_decay



<<<<<<< HEAD
=======
# python test_al_gl.py --dataset isolet --metric raw --K 20 --config config.yaml --resultsdir results_decay 
# python accuracy_al_gl.py --dataset isolet --metric raw --config config.yaml --resultsdir results_decay
# python compile_summary.py --dataset isolet --resultsdir results_decay


>>>>>>> 2d83352046eb53e2a8e4c9cf3c0d3a4a26fb3a91
# python test_al_gl.py --dataset box --metric raw --K 8 --config config_small.yaml --resultsdir results_decay 
# python accuracy_al_gl.py --dataset box --metric raw --config config_small.yaml --resultsdir results_decay
# python compile_summary.py --dataset box --resultsdir results_decay

# python test_al_gl.py --dataset blobs --metric raw --K 16 --config config_small.yaml --resultsdir results_decay 
# python accuracy_al_gl.py --dataset blobs --metric raw --config config_small.yaml --resultsdir results_decay
# python compile_summary.py --dataset blobs --resultsdir results_decay

# python test_al_gl.py --dataset manyblobs --metric raw --K 40 --config config_small.yaml --resultsdir results_decay 
# python accuracy_al_gl.py --dataset manyblobs --metric raw --config config_small.yaml --resultsdir results_decay
# python compile_summary.py --dataset manyblobs --resultsdir results_decay


# python test_al_gl.py --dataset mnistimb-mod3 --K 30 --config config.yaml --resultsdir results_decay 
# python accuracy_al_gl.py --dataset mnistimb-mod3 --config config.yaml --resultsdir results_decay
# python compile_summary.py --dataset mnistimb-mod3 --resultsdir results_decay

# python test_al_gl.py --dataset fashionmnistimb-mod3 --K 30 --config config.yaml --resultsdir results_decay 
# python accuracy_al_gl.py --dataset fashionmnistimb-mod3 --config config.yaml --resultsdir results_decay
# python compile_summary.py --dataset fashionmnistimb-mod3 --resultsdir results_decay

# python test_al_gl.py --dataset emnistvcd --iters 400 --K 120 --config config.yaml --resultsdir results_decay 
# python accuracy_al_gl.py --dataset emnistvcd --iters 400 --config config.yaml --resultsdir results_decay
# python compile_summary.py --dataset emnistvcd --iters 400 --resultsdir results_decay



# python test_al_gl_isolet.py --dataset isolet --metric raw --K 50 --config config_isolet.yaml --resultsdir results_isolet 
# python accuracy_al_gl_isolet.py --dataset isolet --metric raw --config config_isolet.yaml --resultsdir results_isolet
# python compile_summary.py --dataset isolet --resultsdir results_isolet


# python test_al_gl.py --dataset mnist-mod3 --K 20 --config config.yaml --resultsdir results_kde 
# python accuracy_al_gl.py --dataset mnist-mod3 --config config.yaml --resultsdir results_kde
# python compile_summary.py --dataset mnist-mod3 --resultsdir results_kde

# python test_al_gl.py --dataset fashionmnist-mod3 --K 20 --config config.yaml --resultsdir results_kde 
# python accuracy_al_gl.py --dataset fashionmnist-mod3 --config config.yaml --resultsdir results_kde
# python compile_summary.py --dataset fashionmnist-mod3 --resultsdir results_kde

# python test_al_gl.py --dataset emnist-mod5 --iters 400 --K 94 --config config.yaml --resultsdir results_kde 
# python accuracy_al_gl.py --dataset emnist-mod5 --iters 400 --config config.yaml --resultsdir results_kde
# python compile_summary.py --dataset emnist-mod5 --iters 400 --resultsdir results_kde



# python test_al_gl_voptfull.py --dataset mnist-mod3 --config config.yaml --resultsdir results_kde 
# python accuracy_al_gl.py --dataset mnist-mod3 --config config.yaml --resultsdir results_kde
# python compile_summary.py --dataset mnist-mod3 --resultsdir results_kde

# python test_al_gl_voptfull.py --dataset fashionmnist-mod3 --config config.yaml --resultsdir results_kde 
# python accuracy_al_gl.py --dataset fashionmnist-mod3 --config config.yaml --resultsdir results_kde
# python compile_summary.py --dataset fashionmnist-mod3 --resultsdir results_kde

#python test_al_gl_voptfull.py --dataset mnistimb-mod3 --config config.yaml --resultsdir results_kde 
#python accuracy_al_gl.py --dataset mnistimb-mod3 --config config.yaml --resultsdir results_kde
#python compile_summary.py --dataset mnistimb-mod3 --resultsdir results_kde

#python test_al_gl_voptfull.py --dataset fashionmnistimb-mod3 --config config.yaml --resultsdir results_kde 
#python accuracy_al_gl.py --dataset fashionmnistimb-mod3 --config config.yaml --resultsdir results_kde
#python compile_summary.py --dataset fashionmnistimb-mod3 --resultsdir results_kde



