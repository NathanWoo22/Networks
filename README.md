# Discussion    

# Things to do
- Change histogram labels
- Change color of training model
- Send Eric plots


# Running
To activate this conda environment, use    

$ conda activate dnn   

To deactivate an environment, use

$ conda deactivate          

Give your screen a name:
screen -S <name> 

Run you neural network:
tensorman =ml-env run --gpu python -- ./network.py

Or better yet make dnn-env the default:
tensorman default =ml-env gpu
tensorman run --gpu python -- ./network.py

Then detach from your screen with
ctrl + a + d

To reattach or list screen just type
screen -r

To test a network that is currently training:
python generate_plots.py current_model_data current_model_plots

To only use a specific GPU:
os.environ["CUDA_VISIBLE_DEVICES"]="1"