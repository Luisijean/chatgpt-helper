import pandas as pd
import difflib
from collections import abc
from collections import defaultdict

# Fonction pour aplatir les dictionnaires imbriqués de manière récursive
def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                items.extend(flatten_dict({f'{k}[{i}]': item}, parent_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Fonction pour aplatir une liste de dictionnaires
def flatten_list_of_dicts(data):
    return [flatten_dict(d) for d in data]

# Fonction pour trouver les correspondances de champs en utilisant un échantillon
def match_fields(api1_data, api2_data, sample_size=1000, cutoff=0.6):
    api1_sample = api1_data[:sample_size]
    api2_sample = api2_data[:sample_size]
    
    api1_flattened = flatten_list_of_dicts(api1_sample)
    api2_flattened = flatten_list_of_dicts(api2_sample)
    
    if not api1_flattened or not api2_flattened:
        return {}
    
    api1_fields = set(api1_flattened[0].keys())
    api2_fields = set(api2_flattened[0].keys())
    
    field_value_matches = defaultdict(lambda: defaultdict(list))
    
    # Collect field values for comparison
    for record in api1_flattened:
        for field, value in record.items():
            field_value_matches['api1'][field].append(value)
    
    for record in api2_flattened:
        for field, value in record.items():
            field_value_matches['api2'][field].append(value)
    
    field_correlations = {}
    
    for field1 in api1_fields:
        best_match = None
        best_match_score = 0
        
        for field2 in api2_fields:
            match_count = sum(1 for v1, v2 in zip(field_value_matches['api1'][field1], field_value_matches['api2'][field2]) if v1 == v2)
            match_score = match_count / sample_size
            
            if match_score > best_match_score:
                best_match = field2
                best_match_score = match_score
        
        if best_match_score > cutoff:
            field_correlations[field1] = best_match
    
    return field_correlations

# Exemple de données API (ajustez les données et la taille de l'échantillon selon vos besoins)
api1_data = [
    {'user': {'id': 1, 'name': 'Alice'}, 'score': 85},
    {'user': {'id': 2, 'name': 'Bob'}, 'score': 90},
    # Ajoutez plus de données pour atteindre 1000+
]

api2_data = [
    {'identifier': 1, 'fullname': 'Alice', 'points': 85},
    {'identifier': 2, 'fullname': 'Bob', 'points': 90},
    # Ajoutez plus de données pour atteindre 1000+
]

# Appel des fonctions pour aplatir les données et trouver les correspondances de champs
matches = match_fields(api1_data, api2_data, sample_size=1000, cutoff=0.6)
print(matches)












import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import abc, defaultdict
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer

# Fonction pour aplatir les dictionnaires imbriqués de manière récursive
def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                items.extend(flatten_dict({f'{k}[{i}]': item}, parent_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Fonction pour aplatir une liste de dictionnaires
def flatten_list_of_dicts(data):
    return [flatten_dict(d) for d in data]

# Fonction pour générer des paires de champs et calculer des caractéristiques
def generate_feature_pairs(api1_flattened, api2_flattened):
    features = []
    labels = []  # Only used if you have labeled data

    api1_fields = set(api1_flattened[0].keys())
    api2_fields = set(api2_flattened[0].keys())
    
    tfidf_vectorizer = TfidfVectorizer()

    for field1, field2 in itertools.product(api1_fields, api2_fields):
        field1_values = [str(record.get(field1, '')) for record in api1_flattened]
        field2_values = [str(record.get(field2, '')) for record in api2_flattened]

        if field1_values and field2_values:
            combined_values = field1_values + field2_values
            tfidf_matrix = tfidf_vectorizer.fit_transform(combined_values)
            cosine_similarity = (tfidf_matrix[:len(field1_values)] @ tfidf_matrix[len(field1_values):].T).mean()
            
            features.append([cosine_similarity])
            # If you have labeled data:
            # labels.append(1 if field1 corresponds to field2 else 0)

    return np.array(features), np.array(labels) if labels else None

# Fonction pour entraîner un modèle de machine learning
def train_model(api1_data, api2_data, sample_size=1000):
    api1_sample = api1_data[:sample_size]
    api2_sample = api2_data[:sample_size]
    
    api1_flattened = flatten_list_of_dicts(api1_sample)
    api2_flattened = flatten_list_of_dicts(api2_sample)
    
    features, labels = generate_feature_pairs(api1_flattened, api2_flattened)
    
    if labels is not None:
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(features, [1] * len(features))  # Assumes all pairs are positive

    return model

# Fonction pour trouver les correspondances de champs en utilisant le modèle
def match_fields_with_model(model, api1_data, api2_data, sample_size=1000):
    api1_sample = api1_data[:sample_size]
    api2_sample = api2_data[:sample_size]
    
    api1_flattened = flatten_list_of_dicts(api1_sample)
    api2_flattened = flatten_list_of_dicts(api2_sample)
    
    features, _ = generate_feature_pairs(api1_flattened, api2_flattened)
    
    predictions = model.predict(features)
    
    api1_fields = set(api1_flattened[0].keys())
    api2_fields = set(api2_flattened[0].keys())
    
    field_correlations = {}
    index = 0
    for field1, field2 in itertools.product(api1_fields, api2_fields):
        if predictions[index] == 1:
            field_correlations[field1] = field2
        index += 1
    
    return field_correlations

# Exemple de données API (ajustez les données et la taille de l'échantillon selon vos besoins)
api1_data = [
    {'user': {'id': 1, 'name': 'Alice'}, 'score': 85},
    {'user': {'id': 2, 'name': 'Bob'}, 'score': 90},
    # Ajoutez plus de données pour atteindre 1000+
]

api2_data = [
    {'identifier': 1, 'fullname': 'Alice', 'points': 85},
    {'identifier': 2, 'fullname': 'Bob', 'points': 90},
    # Ajoutez plus de données pour atteindre 1000+
]

# Entraîner le modèle
model = train_model(api1_data, api2_data, sample_size=1000)

# Utiliser le modèle pour trouver les correspondances de champs
matches = match_fields_with_model(model, api1_data, api2_data, sample_size=1000)
print(matches)