from flask import Flask, render_template, request, g, redirect
from wtforms import (Form, SubmitField,SelectField)
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.externals.joblib import load
import sqlite3
import skill_level
import os

app = Flask(__name__)
database_path = os.path.join(app.root_path, 'db/match_results.db')
model_path = os.path.join(app.root_path, 'model')

teams = ['New Zealand', 'South Africa', 'England', 'Wales',
         'Scotland', 'Australia', 'France', 'Argentina',
         'Ireland', 'Fiji', 'Italy', 'Samoa', 'Japan',
         'Canada', 'Tonga', 'USA',
         'Georgia', 'Russia', 'Romania']

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(database_path)
    return db


# Create user input form
class TeamForm(Form):

    # Create user team selection
    team1 = SelectField(label = 'Team one', choices = teams, default = teams[0])
    team2 = SelectField(label = 'Team two', choices = teams, default = teams[1])

    submit = SubmitField("Predict!")

def model_predict(team1,team2):

    db = get_db()
    latest_stats = pd.read_sql_query('SELECT * from latest_stats', db)

    # Load model and scaler
    model = load_model(model_path+'/model.h5')
    sc = load(model_path+'/std_scaler.bin')

    # Get each team's latest skill level and sigma
    team1_skill = latest_stats['Team skill'].where(latest_stats['Team'] == team1).dropna().values[0]
    team2_skill = latest_stats['Team skill'].where(latest_stats['Team'] == team2).dropna().values[0]
    team1_sigma = latest_stats['Team sigma'].where(latest_stats['Team'] == team1).dropna().values[0]
    team2_sigma = latest_stats['Team sigma'].where(latest_stats['Team'] == team2).dropna().values[0]
    team1_ranking = latest_stats['Team ranking pts'].where(latest_stats['Team'] == team1).dropna().values[0]
    team2_ranking = latest_stats['Team ranking pts'].where(latest_stats['Team'] == team2).dropna().values[0]

    # Calculate win probability
    win_prob = skill_level.win_probability_single(team1_skill, team2_skill, team1_sigma, team2_sigma)

    df = pd.DataFrame(columns=['Win prob', 'Team skill', 'Opp skill', 'Team ranking pts', 'Opposition ranking pts'])
    df.loc[0] = [win_prob, team1_skill, team2_skill, team1_ranking, team2_ranking]

    X = df[['Win prob', 'Team skill', 'Opp skill', 'Team ranking pts', 'Opposition ranking pts']]
    Y = X.values
    Z = sc.transform(Y)
    prediction = model.predict(Z)

    # Results
    html_win_prob = "Win probability for %s is %s %%" % (team1, round(win_prob, 2) * 100)
    html_score = "Predicted score %s - %s" % (int(round(prediction[0][0])), int(round(prediction[0][1])))

    return html_win_prob, html_score

# Home page
@app.route("/", methods=['GET', 'POST'])
def home():

    form = TeamForm(request.form)

    # Get team names and send to model
    if request.method == 'POST' and form.validate():
        team1 = request.form['team1']
        team2 = request.form['team2']
        return render_template('prediction.html', win_prob=model_predict(team1,team2)[0],
                               score=model_predict(team1, team2)[1])

    # Send template information to index.html
    return render_template('index.html', form=form)

# Return to home page from prediction

@app.route('/button', methods=["GET", "POST"])

def button():

    if request.method == "POST":
        return redirect(("/"))

if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)