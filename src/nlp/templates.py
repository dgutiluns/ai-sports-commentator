def event_to_commentary(event):
    if event['event'] == 'pass':
        return f"Player {event['from_player']} passes to Player {event['to_player']}."
    elif event['event'] == 'goal':
        return f"Goal by Player {event['by_player']}!"
    elif event['event'] == 'turnover':
        return f"Turnover from Player {event['from_player']} to Player {event['to_player']}."
    else:
        return f"{event['event'].capitalize()} detected." 