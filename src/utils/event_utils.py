def filter_self_passes(events):
    """
    Filter out self-passes from the events list.
    Args:
        events: List of event dictionaries
    Returns:
        List of event dictionaries with self-passes removed
    """
    return [event for event in events if not (event['event'] == 'pass' and event['from_player'] == event['to_player'])] 