# Music-Generation-Using-GAN
<h4>Here we have implemented a project where it takes different songs as input and produces a similar song to the ones that are given as input.</h4>
<h4>Our model consists of Generator Network which tires to map the input distribution and a Discriminator network that diferenciates between real and fake song.</h4>
<h4>How to run the project:</h4>
.First include all the songs in the Music Files Directory you want to give as input.<br>
.Run the get_notes.py file to get notes of all songs given in Music Files Folder.<br>
.Now run the convert_note_int.py file to convert notes to unique integers and divide them into sequnces of specific length.<br>
.Run the pytorch_training_file.py to train the model.<br>
.Finally run the prediction.py to get music from randomly generated noise.<br>

