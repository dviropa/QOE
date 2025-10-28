# Rating Quality Distortion
# 5 Excellent Imperceptible
# 4 Good Just perceptible, but not annoying
# 3 Fair Perceptible and slightly annoying
# 2 Poor Annoying, but not objectionable
# 1 Bad Very annoying and objectionable

def rank_qoe(ping):
    if ping <= 80:
        return 'Excellent'
    elif 80 < ping <= 120:
        return 'Good'
    elif 120 < ping <= 160:
        return 'Fair'
    elif 160 < ping < 200:
        return 'Poor'
    elif 200 >= ping:
        return 'Bad'
    return 'Error'

def rank_qoe_3_classes(ping):
    if ping <= 120:
        return 'Excellent'
    elif 120 < ping < 160:
        return 'Fair'
    elif 160 <= ping:
        return 'Bad'
    return 'Error'


def rank_impact(impact):
    if impact <= 80:
        return 'Bad'
    elif 80 < impact <= 120:
        return 'Poor'
    elif 120 < impact <= 160:
        return 'Fair'
    elif 160 < impact < 200:
        return 'Good'
    elif 200 >= impact:
        return 'Excellent'
    return 'Error'

def rank_impact_3_classes(impact):
    if impact <= 120:
        return 'Excellent'
    elif 120 < impact < 160:
        return 'Fair'
    elif 160 <= impact:
        return 'Bad'
    return 'Error'

rank_str_to_int_mapping_3_classes = {
    'Excellent': 3,
    'Fair': 2,
    'Bad': 1
}

rank_str_to_int_mapping = {
    'Excellent': 5,
    'Good': 4,
    'Fair': 3,
    'Poor': 2,
    'Bad': 1
}