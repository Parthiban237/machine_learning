import pandas as pd

# Training Dataset
data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

columns = ['Sky', 'AirTemp', 'Humidity', 'Wind',
           'Water', 'Forecast', 'EnjoySport']

df = pd.DataFrame(data, columns=columns)

# Separate concepts and target
concepts = df.iloc[:, :-1].values
target = df.iloc[:, -1].values


def covers(hypothesis, instance):
    """Check whether a hypothesis covers an instance."""
    for h, x in zip(hypothesis, instance):
        if h != '?' and h != x:
            return False
    return True


def more_general_or_equal(h1, h2):
    """Return True if h1 is more general than or equal to h2."""
    for a, b in zip(h1, h2):
        if a != '?' and a != b:
            return False
    return True


def candidate_elimination(concepts, target):
    n_attributes = concepts.shape[1]

    # Most specific hypothesis
    S = ['Ø'] * n_attributes

    # Most general hypothesis
    G = [['?'] * n_attributes]

    for instance, label in zip(concepts, target):

        # Positive example
        if label == 'Yes':

            # Remove hypotheses from G that do not cover positive example
            G = [g for g in G if covers(g, instance)]

            # Generalize S minimally
            for i in range(n_attributes):
                if S[i] == 'Ø':
                    S[i] = instance[i]

                elif S[i] != instance[i]:
                    S[i] = '?'

            # Remove hypotheses from G that are not
            # more general than or equal to S
            G = [
                g for g in G
                if more_general_or_equal(g, S)
            ]

        # Negative example
        elif label == 'No':

            # Remove hypotheses from S that cover negative example
            if covers(S, instance):
                for i in range(n_attributes):
                    if S[i] == '?':
                        # S cannot be specialized uniquely without
                        # considering all possible attribute values.
                        pass

            # Specialize G
            new_G = []

            for g in G:

                if covers(g, instance):

                    for i in range(n_attributes):

                        if g[i] == '?':
                            values = set(concepts[:, i])

                            for value in values:
                                if value != instance[i]:

                                    new_hypothesis = g.copy()
                                    new_hypothesis[i] = value

                                    # It must be at least as general as S
                                    if more_general_or_equal(
                                            new_hypothesis, S):
                                        new_G.append(new_hypothesis)

                        else:
                          new_G.append(g)

                          G = new_G

    # Remove duplicate hypotheses
    unique_G = []
    for g in G:
        if g not in unique_G:
            unique_G.append(g)

    G = unique_G

    return S, G


# Run Candidate Elimination
S, G = candidate_elimination(concepts, target)

# Display results
print("Final Specific Hypothesis (S):")
print(S)

print("\nFinal General Hypothesis (G):")
for g in G:
    print(g)
