I have written code to, 

1. Model Output(3x200, UNet Architecture(PhaseNet))->Flatten(600x1)-> FC(1)-> Loss(Sigmoid).
2. Binary Classification instead of Multi-class segmentation.
3. Test the model's performance according to the threshold. 

The things you might want to try it out @danukaravishan, 

1. Train the model for 50 epochs, and try out for different hyperparameters tuning. [Click this for Hyperparameter Tuning][https://docs.google.com/spreadsheets/d/1Yo9Mq69OHgiyOVPpCnr7NzwhXTMnd97O_3_VZvjJ-tI/edit?usp=sharing]
2. Try different thresholds other than 0.5 to reduce false positives. (Try drawing precision vs recall curves)
3. Add two FC layers(one hidden FC); uncomment necessary code in lastconv1x1 class. Note that this will overfit the dataset, so add LP Regularization or dropout.
4. 
