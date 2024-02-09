import math

def euclidean_distance(vector1, vector2):
       
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same dimension")
    
    
    distance = 0
    for i in range(len(vector1)):
        distance += (vector1[i] - vector2[i])**2
    return math.sqrt(distance)

def manhattan_distance(vector1, vector2):
       
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same dimension")
    

    distance = 0
    for i in range(len(vector1)):
        distance += abs(vector1[i] - vector2[i])
    return distance

def k_nearest_neighbors(train_data, test_instance, k):
   
    distances = []
    for train_instance, label in train_data:
        distance = euclidean_distance(train_instance, test_instance)
        distances.append((distance, label))
   
    distances.sort(key=lambda x: x[0])
   
    neighbors = distances[:k]
   
    class_votes = {}
    for _, label in neighbors:
        class_votes[label] = class_votes.get(label, 0) + 1
    
    return max(class_votes, key=class_votes.get)

def label_encoding(categories):
   
    label_map = {}
    for i, category in enumerate(categories):
        label_map[category] = i
    return label_map

def one_hot_encoding(categories):
    
    unique_categories = list(set(categories))
    encoded_vectors = []
    for category in categories:
        encoded_vector = [0] * len(unique_categories)
        index = unique_categories.index(category)
        encoded_vector[index] = 1
        encoded_vectors.append(encoded_vector)
    return encoded_vectors

# main class
if __name__ == "__main__":
    # Euclidean distance
    vector1 = [1, 2, 3]
    vector2 = [4, 5, 6]
    print("Euclidean distance:", euclidean_distance(vector1, vector2))

    # Manhattan distance
    print("Manhattan distance:", manhattan_distance(vector1, vector2))

    # k-NN classifier
    train_data = [([1, 2], 'A'), ([2, 3], 'B'), ([3, 4], 'A')]
    test_instance = [1.5, 2.5]
    k = 2
    print("Predicted class label:", k_nearest_neighbors(train_data, test_instance, k))

    # Label encoding
    categories = ['red', 'blue', 'green', 'red', 'green']
    print("Label encoding:", label_encoding(categories))

    # One-Hot encoding
    print("One-Hot encoding:", one_hot_encoding(categories))
