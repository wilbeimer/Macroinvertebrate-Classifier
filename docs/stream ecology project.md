### Problem Description

Stream water quality can be estimated through assessing populations of three kinds of fly larvae (ephemeroptera, plecoptera, trichoptera). Manual collection and visual inspection currently provide inputs for algorithms that calculate water quality measures.

Live stream captures are 

H0: An image recognition and counting algorithm could be trained to determine populations of fly larvae from digital images, so that remotely submitted images could generate an instant estimation of water quality.

 ***\[Needs a strong ‘why’\]***

### Project Stages

1. Initialization, level-setting, topic research, resource identification:  
* Cross-training: ecology for data scientists, data science for the ecologists. One or two brown-bag sessions.  
* Find, read, learn from previous research in this area. Distinguish this project.  
  * [https://link.springer.com/article/10.1007/s10661-020-08545-2](https://link.springer.com/article/10.1007/s10661-020-08545-2) (Gaussian vector support machines obtained a percentage of success in the recognition up method of four organisms to the genus level of 97.1 %.)  
  * [https://www.journals.uchicago.edu/doi/abs/10.1899/09-080.1](https://www.journals.uchicago.edu/doi/abs/10.1899/09-080.1) (Naïve Bayes modeling of stacked decision trees is used)  
  * [https://www.sciencedirect.com/science/article/pii/S0048969724030249?via%3Dihub](https://www.sciencedirect.com/science/article/pii/S0048969724030249?via%3Dihub) (Gradient-weighted Class Activation Mapping (Grad-CAM) visualized the morphological features responsible for the classification of the treated species in the CNN models.)  
  * [https://academic.oup.com/sysbio/article/68/6/876/5368535](https://academic.oup.com/sysbio/article/68/6/876/5368535) (a CNN that has been pretrained on a generic image classification task is exposed to the taxonomic images of interest, and information about its perception of those images is used in training a simpler, dedicated identification system)  
  * [https://ieeexplore.ieee.org/abstract/document/9745134](https://ieeexplore.ieee.org/abstract/document/9745134) (We also presented the comparative analysis results of SSD MobileNET, YoloV4, and Faster R-CNN InceptionV3 deep learning methods and adapting processes for order-level insect classification.)

* Identify data curation needs, such as labeling, augmentation, storage  
* Identify computational needs (i.e. cloud or gpu?)  
* Consensus on data labeling–what attributes should be captured for future use, including follow-on projects?  
* Accuracy measures–what does ‘good enough’ look like, how well do we understand the human error rate, what are we measuring the accuracy of exactly?  
* Identify multiple ML and NN techniques as possible pipeline candidates

2. Data Collection: This includes the curation of all training and test data.   
* Lab  
* Internet  
* ‘Live’, stream collections

3. Image processing: This stage turns ‘images’ into ‘data.’ 

4. Pipeline creation: Coding of candidate methods, train/test split

5. Training and testing: 

6. Evaluation, model choice: Accuracy from ground truth data. New data collections and comparison again with human ground truth.

7. Write-up: A cross-discipline results paper, a second paper on methodology if anything substantive is learned here.

8. Deployment decision: What does this mean? Depending on results, and assuming acceptable accuracy, who is the audience?

9. Future work

How much of a difference is there between dead and live bugs  
How many pictures do we need to take–need to find good data augmentation methods  
How much variation within a species is there–size, color  
How many species do we want to classify  
Is there a quality threshold for images  
Retain metadata about any live captures

Ask Nate/Maria: Canon 5d, DSLR hi-resolution camera and a tripod, what would the resolution be?  
Shoot in raw versus jpeg compression  
2 SD cards: 64Gig each

Next steps: get camera from HIVE  
Take pictures  
Research about data curation and pre-processing  
var11@su.edu