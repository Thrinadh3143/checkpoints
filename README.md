# SAM2 - Segment Anything and Modeling

## Description
SAM2 is a versatile tool designed to perform advanced segmentation and modeling tasks on a wide variety of datasets. Built with state-of-the-art algorithms, SAM2 allows users to easily process, analyze, and visualize their data for efficient modeling and decision-making.

### Key Features:
- Universal segmentation for images, videos, and 3D models.
- Integration with deep learning frameworks.
- High-performance visualization tools.

---

## Installation

### Dependencies:
Ensure the following are installed:
- python>=3.10
- torch>=2.3.1
- torchvision>=0.18.1

### Setup:
1. Clone the SAM2 repository:
   
           git clone https://github.com/username/sam2.git or you can copy it from ./projects/ledigjon/imagesegmentation/SAM2/sam2

2. Installing the SAM2
   
           cd sam2
           pip install -e.
   
4. Download the model checkpoints
  
          cd checkpoints
          ./download_ckpts.sh
          cd ..

## Run
- Execute the following commands:

- Load the cuda Module and verify the version by executing the below commands:

      module load cuda
      nvcc --version

- Run the python script using SAM2 Model:
  
      module load python3.xx
      python --version   
      source <path to SAM2 repository>/my_env/bin/activate
      srun --partition=gpu --gpus-per-node=tesla_v100s:1 --pty python <your_sample_script_path>/your_sample_sam2.py

- After finishing the job deactivate the python environment

      deactivate

### Note:

- if you are running sam2 on your loacl machine, don't need to use "srun --partition=gpu" command, just use the python <your_python_script_path>/script.py
- if you are using clipper then you need to use "srun --partition=gpu" command.


### Source 

- sample python script in the below folder

         ./projects/ledigjon/imagesegmentation/SAM2/Code/

- input image or video source folder

         ./projects/ledigjon/imagesegmentation/SAM2/Input






  

  
   
   
