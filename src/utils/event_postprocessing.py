def deduplicate_events(events):
    deduped = []
    last_dribble = None
    for event in events:
        if event['event'] == 'dribble':
            # Only add if it's a new dribble (different player or not consecutive)
            if (last_dribble is None or
                event['by_player'] != last_dribble['by_player'] or
                event['frame'] > last_dribble['frame'] + 1):
                deduped.append(event)
            last_dribble = event
        else:
            deduped.append(event)
    return deduped 