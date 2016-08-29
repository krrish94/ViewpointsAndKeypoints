[didn't work well. didn't seem to converge at all]

trainTest_old.prototxt -> adapted Shubham's architechture (prototxt) for regression
deploy_old.prototxt    -> corresponding deployment version
solver_old.prototxt	   -> corresponding solver file

------------------------------------------------------------------------------------

[trying out]

trainTest.prototxt     -> adapted VGG-19
deploy.prototxt    	   -> corresponding deployment version

This didn't work well too. Specifically, we tried predicting the raw angle (in degrees) 
in the range [0, 359]. Even after a lot of training, the net seems to predict values 
close to 20 or so. We then hypothesized that this behaviour could be due to the fact that 
all batches were randomly sampled, and had random labels. So, we tried sorting detections 
in the increasing order of their azimuths and retrained the network. Still the behavior 
remains the same. So, we're trying out predictions in radians.
