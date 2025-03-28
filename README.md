# AudioFewShotAST: Few-Shot Learning with Audio Spectrogram Transformer

## Project Overview
AudioFewShotAST is a few-shot learning task using **Audio Spectrogram Transformer (AST)** as the backbone and **Prototypical Networks (ProtoNet)** for audio classification. It is trained on the **Speech Commands dataset**, aiming to perform classification on new, unseen audio commands with few training samples.

## Project Idea and Thought Process
The core idea of this project is to leverage pre-trained AST models together with few-shot learning techniques to build a robust audio classification system. Our approach includes:
- **Data Preprocessing and Normalization:**  
  We use a dedicated script, `compute_norm_params.py`, to compute global normalization parameters (mean and standard deviation) from the training data. These parameters are then used to normalize the dataset consistently during training. This global normalization helps:
  - **Improve Training Stability:** By scaling all data samples to a similar range, the training process becomes more stable and converges faster.
  - **Maintain Relative Scale Information:** Unlike per-sample normalization—which forces each sample to have zero mean and unit variance—global normalization preserves the inherent differences in energy or amplitude across samples, which can be critical for distinguishing between different audio classes.
- **Model Architecture:**  
  The project integrates a pre-trained AST model with a projection layer to generate embeddings, which are then used in a Prototypical Network framework to compute class prototypes and distances for classification.
- **Few-Shot Learning Strategy:**  
  We sample episodic tasks during training, where each episode consists of a small support set and a query set. The model learns to classify query samples based on their distances to the prototypes computed from the support set.
- **Flexibility in Normalization:**  
  While we employ global normalization using the computed parameters, an alternative approach is available (commented in the code) that performs per-sample normalization. This option normalizes each sample to zero mean and unit variance but may discard valuable inter-sample information.

## Project Structure

- `data_loader.py`: Downloads and preprocesses the Speech Commands dataset.
- `model.py`: Defines the AST backbone and Prototypical Network model.
- `train.py`: Training script that implements episodic few-shot learning.
- `test.py`: Testing script to evaluate the model's performance.
- `run.sh`: A shell script to run the training process with a single command.
- `utils.py`: Contains utility functions used during training and evaluation.
- `compute_norm_params.py`: A script to compute global normalization parameters (mean and standard deviation) from the Speech Commands dataset.
- `./data/`: Directory where the Speech Commands dataset, pre-trained models, and the final model are stored.
- `./log/`: Directory where training logs and testing logs are saved.

## Setup Instructions
1. Clone the repository:
    ```bash
    git clone https://github.com/strayberry/AudioFewShotAST.git
    cd AudioFewShotAST
    ```

2. Create and activate the Conda environment from `env.yml`:
    ```bash
    conda env create -f env.yml
    conda activate audiofewshotast
    ```

3. Install any additional dependencies (if needed):
    ```bash
    pip install -r requirements.txt
    ```

4. (Optional) Compute Global Normalization Parameters:
    The `compute_norm_params.py` script computes the global mean and standard deviation from the entire Speech Commands training set. You can run it as follows:
    ```bash
    python compute_norm_params.py
    ```
    The computed parameters can then be used in `data_loader.py` for consistent data normalization.

5. Start training:
    After setting up the environment and installing the dependencies, you can start the training process:
    ```bash
    python train.py
    ```

6. Testing the model:
    Once training is complete, you can evaluate the model using the test script:
    ```bash
    python test.py
    ```

## Notes

- The pre-trained **MIT/ast-finetuned-speech-commands-v2** model is used. You can change the model name in the `model.py` file if needed.
- The **Speech Commands dataset** will be downloaded automatically the first time you run the training script. Ensure that your environment has internet access.
- If you encounter any issues related to package versions or compatibility, make sure that the correct versions of PyTorch, torchaudio, and other dependencies are installed as specified above.
