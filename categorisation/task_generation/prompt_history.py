
## original prompt used on GPT-3
instructions = 'A classification problem consists of a list of input-target pairs.\
                Each input is a vector of length 2, each entry takes continuous values between 0 and 1. \
                The target is a function of the input vector and takes values of either A or B. \
                Please generate 10 input-target pairs:'

## output in the form of tuples (doesn't work)
instructions = 'A classification problem consists of a list of input-target pairs.\
                Each input is a vector of length 2, each entry takes continuous values between 0 and 1. \
                The target is a function of the input vector and takes values of either A or B. \
                Please generate 100 example input-target pairs in the form of tuples:'


## WORKING PROMPT, which provides some novel examples
instructions = 'A classification problem consists of a list of input-target pairs. Each input is a vector of length 2, each entry takes continuous values between 0 and 1. The target is a function of the input vector and takes values of either A or B. Please generate 10 new input-target pairs in the same format as the example. Here are the input-target pairs for an example classification problem: 1. Input: [0.5, 0.5], Target: A; 2. Input: [0.35, 0.31], Target: B; 3. Input: [0.12, 0.45], Target: B; 10. Input: [0.23, 0.46], Target: A; Your output: 1. Input:' # 3. Input: [0.34, 0.56], Target: B;'

# same as above but with sampled random numbers provided as input pairs
instructions = f'A classification problem consists of a list of input-target pairs. Each input is a vector of length 2, each entry takes continuous values between 0 and 1. The target is a function of the input vector and takes values of either A or B. Please generate 10 new input-target pairs in the same format as the example. Here are the input-target pairs for an example classification problem: 1. Input: [{round(np.random.rand(1)[0],2)}, {round(np.random.rand(1)[0],2)}], Target: A; 2. Input: [{round(np.random.rand(1)[0],2)}, {round(np.random.rand(1)[0],2)}], Target: B; 3. Input: [{round(np.random.rand(1)[0],2)}, {round(np.random.rand(1)[0],2)}], Target: B; 10. Input: [{round(np.random.rand(1)[0],2)}, {round(np.random.rand(1)[0],2)}], Target: A; Your output: 1. Input:' # 3. Input: [0.34, 0.56], Target: B;'

# reordered the text a bit to see if I reduce the repeats in the sample (does it help?)
# tried explicity adding do not include input-output pairs from the example did not help
instructions = f'A classification problem consists of a list of input-target pairs. Each input is a vector of length 2, each entry takes continuous values between 0 and 1. The target is a function of the input vector and takes values of either A or B. Here are the input-target pairs for an example classification problem: 1. Input: [{round(np.random.rand(1)[0],2)}, {round(np.random.rand(1)[0],2)}], Target: A; 2. Input: [{round(np.random.rand(1)[0],2)}, {round(np.random.rand(1)[0],2)}], Target: B; 3. Input: [{round(np.random.rand(1)[0],2)}, {round(np.random.rand(1)[0],2)}], Target: B; 10. Input: [{round(np.random.rand(1)[0],2)}, {round(np.random.rand(1)[0],2)}], Target: A; Please generate 10 new input-target pairs in the same format as the example. Your output: 1. Input:' # 3. Input: [0.34, 0.56], Target: B;'

reword a bit but still repeats the samples
instructions = f'A classification problem consists of input-target pairs. Each input is a list of length 2, each entry takes continuous values between 0 and 1. The target is a function of the input list and takes values of either A or B. Here are some input-target pairs for an example classification problem: 1. Input: [{round(np.random.rand(1)[0],2)}, {round(np.random.rand(1)[0],2)}], Target: A; 2. Input: [{round(np.random.rand(1)[0],2)}, {round(np.random.rand(1)[0],2)}], Target: B; 3. Input: [{round(np.random.rand(1)[0],2)}, {round(np.random.rand(1)[0],2)}], Target: B; 10. Input: [{round(np.random.rand(1)[0],2)}, {round(np.random.rand(1)[0],2)}], Target: A; Please generate 10 input-target pairs for a new classification problem in the same format as the example. Your output: 1. Input:' # 3. Input: [0.34, 0.56], Target: B;'

random_samples = np.random.rand(8).round(2)
instructions = f'A classification problem consists of a list of input-target pairs.\
                    Each input is a list of length 2, each entry takes continuous values between 0 and 1.\
                    The target is a function of the input vector and takes values of either A or B.\
                    Here are some input-target pairs for an example classification problem: \
                    1. Input: [{random_samples[0]}, {random_samples[1]}], Target: A;\
                    2. Input: [{random_samples[2]}, {random_samples[3]}], Target: B;\
                    3. Input: [{random_samples[4]}, {random_samples[5]}], Target: B;\
                    10. Input: [{random_samples[6]}, {random_samples[7]}], Target: A;\
                    Please generate 10 new input-target pairs in the same format as the example.\
                    Your output:\
                    1. Input:\
                '

# reformat above the way below doesn't seem yield the same results
instructions = """
                A classification problem consists of a list of input-target pairs. 
                Each input is a vector of length 2, each entry takes continuous values between 0 and 1. 
                The target is a function of the input vector and takes values of either A or B. 
                Please generate 10 new input-target pairs in the same format as the example.
                Here are the input-target pairs for an example classification problem: 
                1. Input: [0.5, 0.5], Target: A; 
                2. Input: [0.35, 0.31], Target: B; 
                3. Input: [0.12, 0.45], Target: B; 
                10. Input: [0.23, 0.46], Target: A; 
                Your output: 
                1. Input: 
                """
                3. Input: [0.34, 0.56], Target: B;'
`` 

# This one interestingly generates code for generate data points
instructions = """ 
                A classification problem consists of a list of input-target pairs. \
                Each input, x, is a vector of length 2 x=[x1, x2], each entry takes continuous values between 0 and 1. \
                The target is a function of the input vector and takes values of either y=A or y=B. \
                Here are the input-target pairs from an example classification problem delimited by triple backticks: 
                ```
                   1. x= [
                   2. Input: [0.35, 0.31], Target: B; 
                   3. Input: [0.12, 0.45], Target: B; 
                   …
                   10. Input: [0.23, 0.46], Target: A;
                ```
                Please generate 10 new input-target pairs in the same format as the example. 
                Your output: 
                x=[
                """

## playing with Julian
instructions = f"A classification problem consists of a list of input-target pairs."\
                " Each input, x, is a vector of length 2 x=[x1, x2], each entry takes continuous values between 0 and 1."\
                " The target, y, is a function of the input vector and takes values of either y=A or y=B.\n\n"\
                "The following is 10 generated input-target pairs:\n"\
                "x=["

# generates new datapoints consistently for 2 dimensional input features and two-category (65B model with temperature 1.) 
# closes it in triple backticks but it doesn't work well for 7B model
instructions = f"Each classification problem consists of a collection of input-target pairs."\
                     " Each input, x, is a vector of length 2, x=[x1, x2], containing inputs features, with each feature taking continuous values between 0 and 1."\
                     " The target, y, is a function of the input vector and takes values of either y=A or y=B.\n\n"\
                     " The following are 10 generated input-target pairs for one such classification problem enclosed within triple backtics:\n"\
                     "```x=["

# using it to generate 10000 tasks
instructions = f"A classification problem consists of a set of input-target pairs."\
                        f" Each input, x, is a vector of length {num_dim}, x = [x1, x2, x3], containing feature values that range continuously between 0 and 1."\
                        " The target, y, is a function of the input vector and can take on values of either y = A or y = B.\n\n"\
                        f" The following are {num_data} input-target pairs generated for one such classification problem:\n"\
                        "x=["