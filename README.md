# dogbreed
This project is based on the Dog-Prediction Kaggle challenge. The objective was the deploy a deep learning model trained with Kaggle data with
an inference server using Flask.
Nginx was used as reverse proxy and load-balancer and Gunicorn was used as an interface between Nginx and the Flask workers runing the model.
The model dogbreeds_model_v7.h5 is based on Inception v3 for the convolution part, a custom fully connected network has been trained to replace the original fully connected layer.
It reached an accuracy of 80% which could have improved by adding some extra-regularization in fully connected layer part (80% of accuracy on validation set 
and 95% of accuracy on Training set).  
Since only 10,000 images were available to predict 120 classes, maybe a single SVM instead of FCN would have done a better job.  
If you wish to use Nginx, you need to install it and update its configuration file located in /etc/nginx/nginx.conf. 
