#python test.py --dataroot datasets/(F-N-0.8)-(F-N-0.6)/trainA --name (F-N-0.8)-(F-W-1.0)-FP-Cyclegan --model test --no_dropout
#测试时需要将checkpoint中的latest_net_G_A.pth换成latest_net_G.pth,来生成假的B.
#fidelity --gpu 0 --isc --input1 results/1/test_latest/images
#python -m pytorch_fid results/FID/(F-N-0.8)-(F-N-1.0)-FP-Cyclegan/original-images results/FID/(F-N-0.8)-(F-W-1.0)-FP-Cyclegan/images --device cuda:0
#python -m pytorch_fid results/FID/original-B results/FID/ni-1-2/images --device cuda:0
#python -m pytorch_fid datasets/(F-N-0.8)-(F-N-0.6)/trainB datasets/(F-N-0.8)-(F-N-0.6)/trainA --device cuda:0
#fidelity --gpu 0 --kid --input1 results/FID/bu-1-2/original-images --input2 results/FID/bu-1-2/images --kid-subset-size 896