from state import State

scenario0 = {
    'rewards': [State(7, 7)],
    'sinks': [State(4, 3)]
}

scenario1 = {    
    'rewards': [State(11, 0), State(13, 0), State(7, 7)],
    'sinks': [State(1, 1), State(1, 0), State(2, 0), State(2, 1), State(2, 2), State(3, 1),
    State(1, 2), State(3, 2), State(4, 1), State(4, 0), State(3, 0), State(2, 0), State(4, 3),
    State(4, 2), State(1,3), State(2, 3), State(3, 3), State(5, 0), State(6, 0), State(7, 0), State(8, 0), State(9, 0),
    State(10, 0)]
}

scenario2 = {
    'rewards': [State(14, 14)],
    'sinks': [State(12, 13), State(13, 12), State(13, 13), State(13, 11), State(11, 13),
    State(12, 11), State(11, 12), State(11, 11), State(7, 7), State(8, 8), State(7, 8), State(8,7),
    State(12, 10), State(10, 12), State(10, 11), State(11, 10)]
}

