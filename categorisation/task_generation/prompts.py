def retrieve_prompt(model, version, num_dim=3, num_data=100, features=None, categories=None):

    instructions = {}

    # llama
    llama_prompt_v0 = f"A classification problem consists of a set of input-target pairs."\
                        f" Each input, x, is a vector of length {str(num_dim)}, x = [x1, x2, x3], containing feature values that range continuously between 0 and 1."\
                        " The target, y, is a function of the input vector and can take on values of either y = A or y = B.\n\n"\
                        f" The following are {str(num_data)} input-target pairs generated for one such classification problem:\n"\
                        "x=["
    
    instructions['llama'] = {}
    instructions['llama']['v0'] = llama_prompt_v0

    # gpt3
    gpt3_prompt_v0 = f"A classification problem consists of a set of input-target pairs."\
                        f" Each input, x, is a vector of length {str(num_dim)}, x = [x1, x2, x3], containing feature values (rounded to 2 decimals) that range continuously between 0 and 1."\
                        " The target, y, is a function of the input vector and can take on values of either y = A or y = B.\n\n"\
                        f" Please generate a list of {str(num_data)} input-target pairs using the following template for each row:\n"\
                        f"- [x1, x2, x3], y"
    
    gpt3_prompt_v1 = f"A categorisation problem consists of a set of input-target pairs."\
                    f" Each input, x, is a vector of length {str(num_dim)}, x = [x1, x2, x3], containing feature values (rounded to 2 decimals) that range continuously between 0 and 1."\
                    " The target, y, is a function of the input vector and can take on values of either y = A or y = B."\
                    " You can choose any naturalistic decision function for the mapping from input to target.  \n\n"\
                    f" Please generate a list of {str(num_data)} input-target pairs for one such categorisation problem using the following template for each row:\n"\
                    f"- [x1, x2, x3], y"
    
    instructions['gpt3'] = {}
    instructions['gpt3']['v0'] = gpt3_prompt_v0
    instructions['gpt3']['v1'] = gpt3_prompt_v1

    # gpt4
    gpt4_prompt_v0 = f"A categorisation problem consists of a set of input-target pairs."\
                    f" Each input, x, is a vector of length {str(num_dim)}, x = [x1, x2, x3], containing feature values (rounded to 2 decimals) that range continuously between 0 and 1."\
                    " The target, y, is a function of the input vector and can take on values of either y = A or y = B.\n\n"\
                    f" Please generate a list of {str(num_data)} input-target pairs for one such categorisation problem using the following template for each row:\n"\
                    f"- [x1, x2, x3], y" ## got code to generate output once but otherwise consistent 
    
    gpt4_prompt_v1 = f"A categorisation problem consists of a set of input-target pairs."\
                    f" Each input, x, is a vector of length {str(num_dim)}, x = [x1, x2, x3], containing feature values (rounded to 2 decimals) that range continuously between 0 and 1."\
                    " The target, y, is a function of the input vector and can take on values of either y = A or y = B."\
                    " You can choose any naturalistic decision function for the mapping from input to target. \n\n"\
                    f" Please generate a list of {str(num_data)} input-target pairs for one such categorisation problem using the following template for each row:\n"\
                    f"- [x1, x2, x3], y"\
                    f" Do not generate any text but just provide the input-target pairs." ## moved this line for pre-prompt
    
    gpt4_prompt_v2 = f"A categorisation problem consists of a set of input-target pairs."\
                    f" Each input, x, is a vector of length {str(num_dim)}, x = [x1, x2, x3], containing feature values (rounded to 2 decimals) that range continuously between 0 and 1."\
                    " The target, y, is a function of the input vector and can take on values of either y = A or y = B."\
                    " For the mapping from input to target, a wide range of naturalistic decision functions can be chosen."\
                    " These functions may encompass complex mathematical operations, linear or non-linear functions, or arbitrary rule-based systems."\
                    " The selected function should be representative of patterns or rules that may exist in real-world categorization tasks. \n\n" \
                    f" Please generate a list of {str(num_data)} input-target pairs for one such categorisation problem using the following template for each row:\n"\
                    f"- [x1, x2, x3], y"
                    
    gpt4_prompt_v3 = f" I am a psychologist who wants to run a category learning experiment."\
                    " For a category learning experiment, I need a list of objects and their category labels."\
                    f" Each object is characterized by three distinct features: shape, size, and colour."\
                    " These feature values (rounded to 2 decimals) range continuously between 0 and 1."\
                    " Each feature should follow a distribution that describes the values they take in the real world. "\
                    " The category label can take the values A or B and should be predictable from the feature values of the object."\
                    " For the mapping from object features to the category label, you can choose any naturalistic function that is"\
                    " representative of patterns or rules that may exist in real-world tasks. \n\n"\
                    f" Please generate a list of {str(num_data)} objects with their feature values and their corresponding"\
                    " category labels using the following template for each row: \n"\
                    "-  feature value 1, feature value 2, feature value 3, category label \n"
    
    instructions['gpt4'] = {}
    instructions['gpt4']['v0'] = gpt4_prompt_v0
    instructions['gpt4']['v1'] = gpt4_prompt_v1
    instructions['gpt4']['v2'] = gpt4_prompt_v2
    instructions['gpt4']['v3'] = gpt4_prompt_v3


    # claude
    features = ['shape', 'size', 'color'] if features is None else features
    categories = ['A', 'B'] if categories is None else categories
    features = [f.lower() for f in features] # make all elements lower case
    categories = [c.lower() for c in categories] # make all elements lower case

    claude_prompt_v0 = f" I am a psychologist who wants to run a category learning experiment."\
                    " For a category learning experiment, I need a list of objects and their category labels."\
                    f" Each object is characterized by three distinct features: {features[0]}, {features[1]}, and {features[2]}."\
                    " These feature values (rounded to 2 decimals) range continuously between 0 and 1."\
                    " Each feature should follow a distribution that describes the values they take in the real world. "\
                    " The category label can take the values A or B and should be predictable from the feature values of the object."\
                    " For the mapping from object features to the category label, you can choose any naturalistic function that is"\
                    " representative of patterns or rules that may exist in real-world tasks. \n\n"\
                    f" Please generate a list of {str(num_data)} objects with their feature values and their corresponding"\
                    " category labels using the following template for each row: \n"\
                    "-  feature value 1, feature value 2, feature value 3, category label \n"

    claude_prompt_v1 = f" I am a psychologist who wants to run a category learning experiment."\
                    " For a category learning experiment, I need a list of objects and their category labels."\
                    f" Each object is characterized by three distinct features: {features[0]}, {features[1]}, and {features[2]}."\
                    " These feature values (rounded to 2 decimals) range continuously between 0 and 1."\
                    " Each feature should follow a distribution that describes the values they take in the real world. "\
                    " The category label can take the values A or B and should be predictable from the feature values of the object."\
                    " \n\n"\
                    f" Please generate a list of {str(num_data)} objects with their feature values and their corresponding"\
                    " category labels using the following template for each row: \n"\
                    "-  feature value 1, feature value 2, feature value 3, category label \n"

    claude_prompt_v2 = f" I am a psychologist who wants to run a category learning experiment."\
                    " For a category learning experiment, I need a list of stimuli and their category labels."\
                    f" Each stimulus is characterized by three distinct features: {features[0]}, {features[1]}, and {features[2]}."\
                     " These feature values (rounded to 2 decimals) range continuously between 0. and 1. where 0. indicates the minimum possible value and 1. the maximum possible value."\
                    f" The category label can be {categories[0]} or {categories[1]} and should be predictable from the feature values of the stimulus."\
                    " \n\n"\
                    f" Please generate a list of {str(num_data)} stimuli with their feature values and their corresponding"\
                    " category labels using the following template for each row: \n"\
                    "-  feature value 1, feature value 2, feature value 3, category label \n"
                    # " Each feature should follow a distribution that describes the values they take in the real world. "\
    
    claude_prompt_v3 = f" I am a psychologist who wants to run a category learning experiment."\
                    " For a category learning experiment, I need a list of stimuli and their category labels."\
                    f" Each stimulus is characterized by three distinct features: {features[0]}, {features[1]}, and {features[2]}."\
                    f" The category label can be {categories[0]} or {categories[1]} and should be predictable from the feature values of the stimulus."\
                    " \n\n"\
                    f" Please generate a list of {str(num_data)} stimuli with their feature values and their corresponding"\
                    " category labels using the following template for each row: \n"\
                    " feature value 1, feature value 2, feature value 3, category label \n"

    claude_prompt_v4 = f" I am a psychologist who wants to run a category learning experiment."\
                    " For a category learning experiment, I need a list of stimuli and their category labels."\
                    f" Each stimulus is characterized by three distinct features: {features[0]}, {features[1]}, and {features[2]}."\
                    " These features can take only numerical values."\
                    f" The category label can be {categories[0]} or {categories[1]} and should be predictable from the feature values of the stimulus."\
                    " \n\n"\
                    f" Please generate a list of {str(num_data)} stimuli with their feature values and their corresponding"\
                    " category labels using the following template for each row: \n"\
                    "-  feature value 1, feature value 2, feature value 3, category label \n"
    
    num_to_text = {2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight'}
    def featurenames_to_text(features, num_dim):
        if num_dim==3:
            return f'{features[0]}, {features[1]}, and {features[2]}'
        elif num_dim==4:
            return f'{features[0]}, {features[1]}, {features[2]}, {features[3]}'
        elif num_dim==6:
            return f'{features[0]}, {features[1]}, {features[2]}, {features[3]}, {features[4]}, and {features[5]}'
    
    claude_prompt_v5 = f" I am a psychologist who wants to run a category learning experiment."\
                        " For a category learning experiment, I need a list of stimuli and their category labels."\
                        f" Each stimulus is characterized by {num_to_text[num_dim]} distinct features: {featurenames_to_text(features, num_dim)}."\
                        " These features can take only numerical values."\
                        f" The category label can be {categories[0]} or {categories[1]} and should be predictable from the feature values of the stimulus."\
                        " \n\n"\
                        f" Please generate a list of {str(num_data)} stimuli with their feature values and their corresponding"\
                        " category labels sequentially without skipping any row using the following template for each row: \n"\
                        f" 1: feature value 1, feature value 2,..., feature value {str(num_dim)}, category label \n"   

    instructions['claude'] = {}
    instructions['claude']['v0'] = claude_prompt_v0
    instructions['claude']['v1'] = claude_prompt_v1
    instructions['claude']['v2'] = claude_prompt_v2
    instructions['claude']['v3'] = claude_prompt_v3
    instructions['claude']['v4'] = claude_prompt_v4
    instructions['claude']['v5'] = claude_prompt_v5
    
    return instructions[model][version]

def retrieve_tasklabel_prompt(model, version, num_dim=3, num_tasks=100):

    instructions = {}
    
    num_to_text = {2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight'}
    claude_feature_names_prompt_v0 = f" I am a psychologist who wants to run a category learning experiment."\
                    " In a category learning experiment, several objects vary along different feature dimensions, with each object belonging to a category."\
                    " The feature dimensions of an object are such that they can be used to assign a category to an object."\
                    f" Please generate a list of {str(num_tasks)} feature dimensions along which objects could vary: \n"\
                    
    claude_feature_names_prompt_v1 = f" I am a psychologist who wants to run a category learning experiment."\
                    " In a category learning experiment, there are several objects with each object belonging to a single category."\
                    f" Each object takes different values along {str(num_dim)} choosen feature dimensions. "\
                    f" Please generate a list of {str(num_tasks)} pairs of {str(num_dim)} feature dimensions for objects in a category learning experiment and and their category label: \n"\

    claude_feature_names_prompt_v2  = f" I am a psychologist who wants to run a category learning experiment."\
                 " There are several stimuli in a category learning experiment, with each stimulus belonging to one category."\
                f" These stimuli take on different values along the {num_to_text[num_dim]} choosen feature dimensions. "\
                 " The category to which a stimulus belongs to is binary-valued. For example, they can be True/False or 0/1."\
                f" Please generate names for {num_to_text[num_dim]} stimulus feature dimensions and a corresponding category name for {str(num_tasks)} category learning experiments: \n"\
                f"- name of feature dimension 1, name of feature dimension 2, ..., name of feature dimension {str(num_dim)}, name of the category \n"
    
    claude_feature_names_prompt_v3  = f" I am a psychologist who wants to run a category learning experiment."\
                 f" In a category learning experiment, there are many different {num_to_text[num_dim]} dimensional stimuli, with each stimulus belonging to a single category."\
                  " The category to which a stimulus belongs to is binary-valued. For instance, it can be True/False or 0/1."\
                 f" Please generate names for {num_to_text[num_dim]} stimulus feature dimensions and a corresponding category name for {str(num_tasks)} category learning experiments: \n"\
                 f"- name of feature dimension 1, name of feature dimension 2, ..., name of feature dimension {str(num_dim)}, name of the category \n"
    
    claude_feature_names_prompt_v4  = f" I am a psychologist who wants to run a category learning experiment."\
                 f" In a category learning experiment, there are many different {num_to_text[num_dim]} dimensional stimuli, with each stimulus belonging to a single category."\
                  " The category to which stimuli belong to takes on one of two labels."\
                 f" Please generate names for {num_to_text[num_dim]} stimulus feature dimensions and {num_to_text[2]} category labels for {str(num_tasks)} category learning experiments: \n"\
                 f"- feature dimension 1, feature dimension 2, ..., feature dimension {str(num_dim)}, category label 1, category label 2  \n"
    
    claude_feature_names_prompt_v5  = f" I am a psychologist who wants to run a category learning experiment."\
                 f" In a category learning experiment, there are many different {num_to_text[num_dim]}-dimensional stimuli, each of which belongs to one of two possible real-world categories."\
                 f" Please generate names for {num_to_text[num_dim]} stimulus feature dimensions and {num_to_text[2]} corresponding categories for {str(num_tasks)} different category learning experiments: \n"\
                 f"- feature dimension 1, feature dimension 2, ..., feature dimension {str(num_dim)}, category label 1, category label 2  \n"   
    
    instructions['claude'] = {}
    instructions['claude']['v0'] = claude_feature_names_prompt_v0
    instructions['claude']['v1'] = claude_feature_names_prompt_v1
    instructions['claude']['v2'] = claude_feature_names_prompt_v2
    instructions['claude']['v3'] = claude_feature_names_prompt_v3
    instructions['claude']['v4'] = claude_feature_names_prompt_v4
    instructions['claude']['v5'] = claude_feature_names_prompt_v5
    
    return instructions[model][version]

def synthesize_functionlearning_problems(model, version, num_dim=1, num_tasks=100):

    instructions = {}
    features_to_text = {1: 'a real-world feature is mapped to its corresponding target, with both feature and target taking on continuous values', 2: 'two real-world features are mapped to their corresponding target, with features and target taking on continuous values'}
    format_to_text = {1: '- feature name, target name', 2: '- feature name 1, feature name 2, target name'}
    synthesize_feature_names_prompt_v0  = f" I am a psychologist who wants to run a function learning experiment. "\
                 f"In a function learning experiment, {features_to_text[num_dim]}."\
                 f" Please generate names for features and its corresponding target for {str(num_tasks)} different function learning experiments: \n"\
                 f"{format_to_text[num_dim]} \n"  
    
    instructions['claude'] = {}
    instructions['claude']['v0'] = synthesize_feature_names_prompt_v0

    
    return instructions[model][version]

def synthesize_decisionmaking_problems(model, version, num_dim=2, num_tasks=100):

    instructions = {}
    features_to_text = {1: 'a real-world feature is mapped to its corresponding target, with both feature and target taking on continuous values', 2: 'two real-world features are mapped to their corresponding target, with features and target taking on continuous values'}
    format_to_text = {1: '- feature name, target name', 2: '- feature name 1, feature name 2, target name'}
    synthesize_feature_names_prompt_v0  = f" I am a psychologist who wants to run a decision-making experiment. "\
                 f"In a decision-making experiment, {features_to_text[num_dim]}."\
                 f" Please generate names for features and its corresponding target for {str(num_tasks)} different decision-making experiments: \n"\
                 f"{format_to_text[num_dim]} \n"  
    
    instructions['claude'] = {}
    instructions['claude']['v0'] = synthesize_feature_names_prompt_v0

    
    return instructions[model][version]

def generate_data_functionlearning_problems(model, version, num_data=20, num_dim=1, features=None, target=None):

    instructions = {}
    instructions['claude'] = {}
    # feature, target = 'Game difficulty', 'Game engagement'
    generate_data_prompt_v0 = f" I am a psychologist who wants to run a function learning experiment."\
                          " For a function learning experiment, I need a list of feature and target pairs."\
                         f" The feature and target in this case are {features[0].lower()} and {target.lower()} respectively."\
                         f" {features[0].capitalize()} can take only numerical values and must be continuous."\
                         f" {target.capitalize()} should be predictable from the {features[0].lower()} and must also take on continuous values."\
                          " \n\n"\
                        f" Please generate a list of {str(num_data)} feature-target pairs"\
                         " sequentially using the following template for each row: \n"\
                        f" 1: {features[0].lower()} value, {target.lower()} value \n"\
                        f"Please do not skip any row; values taken by {features[0].lower()} and {target.lower()} do not need to be ordered."
    
    generate_data_prompt_v1 = f" I am a psychologist who wants to run a function learning experiment."\
                          " For a function learning experiment, I need a list of feature and target pairs."\
                         f" The feature and target in this case are {features[0].lower()} and {target.lower()} respectively."\
                         f" The feature can take only numerical values and must be continuous."\
                         f" {target.capitalize()} should be predictable from the feature and must also take on continuous values."\
                          " \n\n"\
                        f" Please generate a list of {str(num_data)} feature-target pairs"\
                         " sequentially using the following template for each row: \n"\
                        f" 1: {features[0].lower()} value, {target.lower()} value \n"\
                        f"Please do not skip any row; values taken by features and target do not need to be ordered."
    
    def featurenames_to_text(features, num_dim):
        if num_dim==1:
            return f'feature in this case is {features[0].lower()}', 'The feature takes', '- feature value, target value'
        elif num_dim==2:
            return f'features in this case are {features[0].lower()} and {features[1].lower()}', 'These features take', '- feature value 1, feature value 2, target value'
        
    feature_text, style, template = featurenames_to_text(features, num_dim)
    generate_data_prompt_v2 = f" I am a psychologist who wants to run a function learning experiment."\
                          " For a function learning experiment, I need a list of features with their corresponding target."\
                         f" The {feature_text}."\
                         f" {style} on only numerical values and must be continuous."\
                         f" The target, {target.lower()}, should be predictable from the feature values and must also take on continuous values."\
                          " \n\n"\
                         f" Please generate a list of {str(num_data)} feature-target pairs"\
                          " sequentially using the following template for each row: \n"\
                         f" {template} \n"\
                         f" Please do not skip any row; values taken by features and targets do not need to be ordered."
    
    instructions['claude']['v0'] = generate_data_prompt_v0
    instructions['claude']['v1'] = generate_data_prompt_v1
    instructions['claude']['v2'] = generate_data_prompt_v2
    
    return instructions[model][version]

def generate_data_decisionmaking_problems(model, version, num_dim=2, num_data=20, features=None, target=None):

    instructions = {}
    instructions['claude'] = {}

    # feature, target = 'Game difficulty', 'Game engagement'
    def featurenames_to_text(features, num_dim):
        if num_dim==1:
            return f'feature in this case is {features[0].lower()}', 'The feature takes', '- feature value, target value'
        elif num_dim==2:
            return f'features in this case are {features[0].lower()} and {features[1].lower()}', 'These features take', '- feature value 1, feature value 2, target value'
        
    feature_text, style, template = featurenames_to_text(features, num_dim)
    generate_data_prompt_v0 = f" I am a psychologist who wants to run a decision-making experiment."\
                          " For a decision-making experiment, I need a list of features with their corresponding target."\
                         f" The {feature_text}."\
                         f" {style} on only numerical values and must be continuous."\
                         f" The target, {target.lower()}, should be predictable from the feature values and must also take on continuous values."\
                          " \n\n"\
                         f" Please generate a list of {str(num_data)} feature-target pairs"\
                          " sequentially using the following template for each row: \n"\
                         f" {template} \n"\
                         f"  Please do not skip any row; values taken by features and targets do not need to be ordered."
    
    instructions['claude']['v0'] = generate_data_prompt_v0
    
    return instructions[model][version]
