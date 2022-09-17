# Author: Jeanette Zamora Gonzalez
# Date: 27/06/22
# Project: Train a computer vision model


In the first training, I used transfer learning. Basically, Exp_01 uses the config example.yaml.
I modify the parameters, for example, the data augmentation, learning rate, etc, finally I add the l2 regularization, and I don't get good results. **The accuracy train went about 0.49**. 


After doing multiple experiments, **I used fine-tuning**.
I unfroze the entire model and instead of using the weights from resnet_50, I trained the model entirely.
The best model obtained an **accuracy of 0.91 in train**.
Using the parameters of the said experiment, the model was evaluated with the test data. *Resulting in: Your model accuracy is 0.6490!*


Finally, to obtain better results, Detectron2 was used, whose purpose is image detection. Once our cars and trucks were detected, they were cut out and the model was trained with this new data set.
Use the same parameters with which the best result was obtained with transfer learning and **the result went 0.65 in the model.**
By increasing the dropout and decreasing the learning rate, I arrived at the best model, *with an accuracy of 0.72.*

# Graph:

https://docs.google.com/presentation/d/1v6FAZIXs_L9Kr7-1CyQXx5krto0XuRTov88289uOfbs/edit?usp=sharing


