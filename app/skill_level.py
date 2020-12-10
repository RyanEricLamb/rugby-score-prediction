import trueskill
from trueskill import Rating, quality_1vs1, rate_1vs1
import math
import pandas as pd

draw_prob = 0.02
sig = 4

env = trueskill.TrueSkill(draw_probability=draw_prob)
env.make_as_global()


def win_probability_teams (team1,team2):

    delta_mu = team1.mu - team2.mu
    sum_sigma = (team1.sigma ** 2)+(team2.sigma ** 2)
    size = 2
    denom = math.sqrt(size * (trueskill.BETA * trueskill.BETA) + sum_sigma)
    ts = trueskill.global_env()
    return ts.cdf(delta_mu / denom)


def rate_teams(data):

    # TODO - move this out of the function
    rank_baseline = pd.read_csv('irb_rankings_oct_2003.csv')
    data = data.sort_values(['Match Date'], ascending=True)

    try:
        # Create dict of baseline ranking scores for baseline skill
        h = []
        for team, points in zip(rank_baseline['Teams'], rank_baseline['Points']):
            h.append([team, Rating()])
        var_names = dict(h)

    except Exception as e:
        print(e)
        print("Failed team baseline score assignment")

    # Calculate TrueSkill and win probability
    team = data['Team'].values.tolist()
    opp = data['Opposition'].values.tolist()
    result = data['Result'].values.tolist()
    date = data['Match Date'].values.tolist()

    z = []
    for t, o, r, d in zip(team, opp, result, date):
        if (t in var_names) & (o in var_names):

            if r == 'draw':
                var_names[t], var_names[o] = rate_1vs1(var_names[t], var_names[o], drawn=True)
            else:
                var_names[t], var_names[o] = rate_1vs1(var_names[t], var_names[o])

            # Store win probability for each game
            z.append([t, o, d, win_probability_teams(var_names[t], var_names[o]),
                      var_names[t].mu, var_names[o].mu, var_names[t].sigma, var_names[o].sigma, quality_1vs1(var_names[t], var_names[o])])
    g = []

    for team in rank_baseline['Teams']:
        g.append([team, var_names[team].mu])

    match_probs = pd.DataFrame(z)

    match_probs.columns = ['Team', 'Opposition', 'Match Date', 'Win prob', 'Team skill', 'Opp skill','Team sigma', 'Opp sigma', 'Match Quality']
    match_probs['Match Date'] = pd.to_datetime(match_probs['Match Date'], format='%Y/%m/%d')
    data = pd.merge(data, match_probs, on=(['Team', 'Opposition', 'Match Date']))

    return data



def win_probability_single(team1, team2, team1sigma, team2sigma):

    delta_mu = team1 - team2
    sum_sigma = (team1sigma ** 2)+(team2sigma ** 2)
    size = 2
    denom = math.sqrt(size * (trueskill.BETA * trueskill.BETA) + sum_sigma)
    ts = trueskill.global_env()

    return ts.cdf(delta_mu / denom)
