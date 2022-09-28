# Rose Sales Chatbot System

## How to start

To start the Chatbot, use command: python main.py

To start benchmarking test, use command: python test.py

## Important notes

The Glove model is really large but the performance in this chatbot is very bad. 
Therefore, this zip file does not contain the Glove model and it is also unnecessary. 
However, if you want to try Glove, you can download the Glove from https://uniofnottm-my.sharepoint.com/:f:/g/personal/scylj1_nottingham_ac_uk/ElMdlYIlxa9Bj3ryb8VuYwMBBE1av0sY2GnD7Q2JF-fiLw?e=tQ7GP1. Please unzip two text files to 'data/glove' folder and unzip 'glove_model' to 'model' folder. 
It is strongly not recommended to use it in this chatbot. 

## How to train a model

All the models have been trained already 

To train the sequential machine learning model, use command: python sequence.py

To train the GRU model, use command: python gru.py

To train the GRU with Glove model, use command: python glove.py

## How to use different intent matching approaches

Change the settings in main.py according to the comments