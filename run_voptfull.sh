# python test_al_gl_voptfull.py --dataset mnist-mod3 --config config.yaml --resultsdir results_kde --sopt 1
# python accuracy_al_gl.py --dataset mnist-mod3 --config config.yaml --resultsdir results_kde
# python compile_summary.py --dataset mnist-mod3 --resultsdir results_kde

# python test_al_gl_voptfull.py --dataset fashionmnist-mod3 --config config.yaml --resultsdir results_kde --sopt 1
# python accuracy_al_gl.py --dataset fashionmnist-mod3 --config config.yaml --resultsdir results_kde
# python compile_summary.py --dataset fashionmnist-mod3 --resultsdir results_kde


python test_al_gl_voptfull.py --dataset mnistimb-mod3 --config config.yaml --resultsdir results_kde --sopt 1
python accuracy_al_gl.py --dataset mnistimb-mod3 --config config.yaml --resultsdir results_kde
python compile_summary.py --dataset mnistimb-mod3 --resultsdir results_kde

python test_al_gl_voptfull.py --dataset fashionmnistimb-mod3 --config config.yaml --resultsdir results_kde --sopt 1
python accuracy_al_gl.py --dataset fashionmnistimb-mod3 --config config.yaml --resultsdir results_kde
python compile_summary.py --dataset fashionmnistimb-mod3 --resultsdir results_kde

python test_al_gl.py --dataset emnistvcd --iters 400 --K 120 --config config.yaml --resultsdir results_decay 
python accuracy_al_gl.py --dataset emnistvcd --iters 400 --config config.yaml --resultsdir results_decay
python compile_summary.py --dataset emnistvcd --iters 400 --resultsdir results_decay