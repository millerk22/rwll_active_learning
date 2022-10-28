python test_al_gl.py --dataset box --metric raw --K 8 --config config_toy.yaml --resultsdir results 
python accuracy_al_gl.py --dataset box --metric raw --config config_toy.yaml --resultsdir results
python compile_summary.py --dataset box --resultsdir results

python test_al_gl.py --dataset blobs --metric raw --K 16 --config config_toy.yaml --resultsdir results 
python accuracy_al_gl.py --dataset blobs --metric raw --config config_toy.yaml --resultsdir results
python compile_summary.py --dataset blobs --resultsdir results


