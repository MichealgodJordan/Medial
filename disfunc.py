import torch 

#distance and similarity functions used to value similarity
def L2(X,Y):
    similarity = -torch.cdist(X, X, p=2)
    return similarity

def norm_L2(X, Y):
    square_sum_X = torch.sum(X ** 2, dim=1, keepdim=True)
    square_sum_Y = torch.sum(Y ** 2, dim=0, keepdim=True)
    distances = square_sum_X + square_sum_Y - 2.0 * torch.matmul(X, Y)
    mean = torch.mean(distances)
    std = torch.std(distances)
    normalized_distances = (distances - mean) / std
    similarity = -normalized_distances
    return similarity

def Cor(X, Y):
    X_mean = X.mean(dim=1, keepdim=True)
    X_std = X.std(dim=1, keepdim=True)
    normalized_X = (X - X_mean) / X_std

    Y_mean = Y.mean(dim=0, keepdim=True)
    Y_std = Y.std(dim=0, keepdim=True)
    normalized_Y = (Y - Y_mean) / Y_std
    
    correlation_matrix = torch.matmul(normalized_X, normalized_Y) / 128

    return correlation_matrix

def Manhattan_Distance(X, Y):
    X_expanded = X.unsqueeze(1) - X.unsqueeze(0)
    distance_matrix = torch.sum(torch.abs(X_expanded), dim=2)
    similarity = -distance_matrix
    return similarity

def Cosine(X, Y):
    dot_prod = torch.matmul(X, Y)
    return torch.div(
        dot_prod,
        (torch.norm(X)*torch.norm(Y))
    )
