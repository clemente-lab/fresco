def write_to_file(lines, filepath):
    f = open(filepath, 'w')
    for line in lines:
        f.write(line+"\n")
    f.close()

def feature_output_lines(outcome):
    lines = []
    
    feature_vector = outcome.feature_vector
    
    rec_list = feature_vector.get_record_list()
    populations = [record.get_abundance() for record in rec_list]    
    pred_scores = outcome.feature_scores

    properties = [(rec_list[index].get_id(), rec_list[index].get_scope(), pred_scores[index],
                   populations[index]) for index in range(len(rec_list))] 
    properties.sort(key=lambda prop:prop[2], reverse=True)

    header = ("GROUP_ID", "GROUP_SCOPE", "GROUP_SCORE", "GROUP_ABUNDANCE")
    properties[:0] = [header]

    for prop in properties:
        line = ""
        for i in range(len(prop)):
            line += str(prop[i])
            if i != len(prop) - 1:
                line += "\t"
        lines.append(line)
        
    return lines
    
def outcome_output_lines(outcomes):
    lines = []
    
    header = ("PREDICTION_QUALITY_SCORE",)
    properties = [header]

    for outcome in outcomes:
        properties.append( (outcome.prediction_quality,) ) 

    for prop in properties:
        line = ""
        for i in range(len(prop)):
            line += str(prop[i])
            if i != len(prop) - 1:
                line += "\t"
            lines.append(line)
            
    return lines