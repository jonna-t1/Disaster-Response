# Disaster Response Pipeline Project

### Aim for the project
Analysing thousands of real messages that were sent during natural disasters. Either to social media or through disaster
response organisations. While using a ETL pipeline that processes message and categorical data which are loaded from a 
SQLite database. The machine learning pipeline (NLP) creates a multi-output supervised learning model. 
Then using a web application to extract from the database to provide data visualisations and the machine learning model 
will classify new messages for 36 categories.

During natural disasters it can become difficult to filter out key information which the responder can use which this 
application hopes to address.

### Files in the project
```bash
app
| - template 
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- requirements.txt # requirements for the project 
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py
|- DisasterResponse.db # database to save clean data to
models
|- query_data.py # database query to get an idea of table contents
|- Grid.py # A class that makes model selection easier. Not used in the current version
|- train_classifier.py # script that saves ML classifier
|- classifier.pkl # saved model
README.md
```
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
