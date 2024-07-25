# Organization 
Each folder with have another README.md, here is an overview of each folder.

## current_model_data and current_model_plots
Two folders that can be used to observe progress of model. 

## datasets
Contains datasets (.npz files) for training and testing.

## generate_datasets
Contains python scripts to generate datasets.

## network_results 
The models and results of various trained models.

## optimize_models
A test completed over a long break (possibly Thanksgiving?) used to optimize learning rate. 

## plots
Various plots, mostly from visualize_datasets and plotting

## test_models
Python scripts to test a lot of models at once, for example will test all baseline noise models at once. 

## visualise_datasets
Create plots to see what various datasets look like




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
python testing.py current_model_data current_model_plots Epos_Cropped.npz "Cropped Showers" "Cropped Showers"
python testing.py best_models/convolutional best_models/cross_train Sibyll.npz "EPOS trained, Sibyll test" "EPOS trained, Sibyll test"
python testing.py current_model_data current_model_plots epos_noise_percentage/Epos_20.0.npz "decay_steps=1000, decay_rate=0.96"

To only use a specific GPU:
os.environ["CUDA_VISIBLE_DEVICES"]="1"