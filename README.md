<<<<<<< HEAD
# Image Colorization Starter Code
The objective is to produce color images given grayscale input image. 

## Setup Instructions
Create a conda environment with pytorch, cuda. 

`$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia scikit-image matplotlib pillow pathlib tqdm`

For systems without a dedicated gpu, you may use a CPU version of pytorch.
`$ conda install pytorch torchvision torchaudio cpuonly -c pytorch scikit-image matplotlib pillow pathlib tqdm`

## Dataset
Use the zipfile provided as your dataset. You are expected to split your dataset to create a validation set for initial testing. Your final model can use the entire dataset for training. Note that this model will be evaluated on a test dataset not visible to you.

## Code Guide
Baseline Model: A baseline model is available in `basic_model.py` You may use this model to kickstart this assignment. We use 256 x 256 size images for this problem.
-	Fill in the dataloader, (colorize_data.py)
-	Fill in the loss function and optimizer. (train.py)
-	Complete the training loop, validation loop (train.py)
-	Determine model performance using appropriate metric. Describe your metric and why the metric works for this model? 
- Prepare an inference script that takes as input grayscale image, model path and produces a color image. 

## Additional Tasks 
- The network available in model.py is a very simple network. How would you improve the overall image quality for the above system? (Implement)
- You may also explore different loss functions here.

## Bonus
You are tasked to control the average color/mood of the image that you are colorizing. What are some ideas that come to your mind? (Bonus: Implement)

## Solution
- Document the things you tried, what worked and did not. 
- Update this README.md file to add instructions on how to run your code. (train, inference). 
- Once you are done, zip the code, upload your solution.  
=======
# Image Colorization
The task of colorising greyscale images previously needed significant user inputs, but can nowadays be automated thanks to the recent improvement of deep learning.

![image](https://user-images.githubusercontent.com/90635018/141793231-86618ae2-e885-4583-a644-d4aff83068e2.png)


The objective of this challenge is to colour images given grayscale input image. The dataset contains 4282 images in JPEG format. 
My first approach was simply to work in the RGB colour space. It worked, and the learning process was quite fast. However, the loss of information when passing through smaller layers led to blurred images. 
After some research, I found an article with an interesting approach consisting in working in the LAB colour space. It consist in separating the luminance L from the colour components A and B, and is intended to be perceptually uniform. Then, our model only has to learn how to produce those A and B components, and adding them to the greyscale image. Even if the colourisation is still not so precise, the final image now looks sharp enough. 
The model implemented remains unchanged, except for the fact that it output a 2 dimensional AB image rather that a 3 dimensional RBG image. 
DataLoader
Regarding preprocessing, I do some data augmentation through the RandomCrop and RandomHorizontalFlip transforms. In the _getitem_ fonction, I have to decompose the original image in the LAB colour space. 

Optimizer 
I use the Adam optimizer, which is a widely used improvement of the SGD. 

Metric
I chose to use the MSE loss, it is to say the L2 norm, as it is the standard for image comparison. I tried the L1 norm, but it didn’t improved the result of the algorithm. 
I have to precise that the loss is computed on the difference between the AB images, and not the RGB images.
Inference script
The file predict_color.py can be used to predict the color of a greyscale image using my model.
python predict_color.py -image example_gray.jpg -model trainedModel

Improve the image quality of the model
The use of the LAB colour space improves the quality of the result, without significant changes in the model. Another possibility would be to use the predicted colour of the RGB model, and to adjust it to fit le lightness of the input .

Average mood of the image
The function get_histogram in train.py produces histograms for the three colour channels. An exemple is given below. You can access the histogram of the generated image within the inference script.


python predict_color.py -image example_gray.jpg -model trainedModel -histo True


>>>>>>> 450ae6d961e597d4bccaae43073e75b43b465f79
