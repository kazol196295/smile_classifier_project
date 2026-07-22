

# **Frameworks | Libraries | Deployment Mini Project (Smile Classifier)** 

**Submission Deadline: 3 pm, 23/07/2026** 

1. Download the following dataset: 

<u>https://www.kagpgle.com/datasets/chazzer/smiling-or-not-face-data</u> 

2. Prepare a training script for smile classifier – 2 class, smiling, not smiling (scikit-learn preferred) 

3. Save the trained model in pickle file 

4. Prepare Inference script using the saved model 

5. Prepare a FastAPI project named “Smile Classifier - YourID” 

6. There will be four menu – Home, Train, Classify, History 

7. In the Home page briefly explain how you have prepared the training and inference script for the smile classifier model, which framework and which model you have chosen and why. 

8. In the Train page there will be upload option with multiple image file. 

9. In the Classify page there will be file upload option for uploading an image file. 

10. 

- 11.After uploading the image file, each image file will be converted to jpg if not and saved to file system. 

- 12.For training, use the uploaded files to train and save smile classifier model pickle file and delete the uploaded images. 

- 13.For classification, using the uploaded file do the classification and save the result in database and show the result in another page 

- 14.In the history page there will be a table consisting of Image, Class and DateTime 

- 15.Connect any database you prefer with the project. Save the image path, resultant class and datetime in a table (PostgreSQL preferred) 

- 16.For database connection you can use ORM (SQLAlchemy preferred) 17.Prepare a docker network – YSDTP_B5_AI_YourID 

- 18.Docker compose file for database using same network 


