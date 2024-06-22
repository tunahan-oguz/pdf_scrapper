import torch.utils.data as data
import torch
import numpy as np
from train.dataset import AnatomDataset
from train.image_transform import IMAGE_TRANSFORMS
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, top_k_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def get_embeddings(model, dataloader):
    embeddings = []
    labels = []

    with torch.no_grad():
        for data in dataloader:
            images, lbls = data
            outputs = model(images)
            embeddings.extend(outputs.cpu().numpy())
            labels.extend(lbls.cpu().numpy())

    return np.array(embeddings), np.array(labels)


#config and to_absolute_path?
dataset_root = to_absolute_path(config.paths.dataset_root)
vocab_path = to_absolute_path(config.paths.vocab_path)
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

test_dataset = AnatomDataset(root=dataset_root, split="test", vocab=vocab, transform=IMAGE_TRANSFORMS)
test_dataloader = data.DataLoader(
    dataset=test_dataset, 
    batch_size=config.training.batch_size, 
    collate_fn=AnatomDataset.collate_fn,
    num_workers=config.training.num_workers, 
    pin_memory=True, 
    drop_last=False, 
    shuffle=False
    )

#DONT KNOW HOW YOU INITIALIZE THE MODEL?
model = YourModel()
model.load_state_dict(torch.load('path_to_your_model.pth'))

test_embeddings, test_labels = get_embeddings(model, test_dataloader)

#HOW TO CALCULATE REFERENCE EMBEDDINGS?
reference_embeddings, reference_labels = test_embeddings, test_labels

cos_sim = cosine_similarity(test_embeddings, reference_embeddings)

def evaluate_cosine_similarity(cos_sim, test_labels, reference_labels, k=5):
    recall_at_k = []
    precisions = []
    recalls = []
    f1s = []

    for i in range(len(test_labels)):
        # Get the indices of the top k similar embeddings
        top_k_indices = np.argsort(-cos_sim[i])[:k]
        top_k_labels = reference_labels[top_k_indices]

        # Calculate Recall@K
        if test_labels[i] in top_k_labels:
            recall_at_k.append(1)
        else:
            recall_at_k.append(0)

        # Calculate precision, recall, f1 for binary relevance (for each test instance)
        true_positive = np.sum(top_k_labels == test_labels[i])
        false_positive = k - true_positive
        false_negative = 1 - true_positive

        precision = true_positive / k
        recall = true_positive / 1  # because there's only one true label
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    average_recall_at_k = np.mean(recall_at_k)
    average_precision = np.mean(precisions)
    average_recall = np.mean(recalls)
    average_f1 = np.mean(f1s)

    print(f'Recall at {k}: {average_recall_at_k:.2f}')
    print(f'Precision: {average_precision:.2f}')
    print(f'Recall: {average_recall:.2f}')
    print(f'F1-score: {average_f1:.2f}')

# Evaluate the model
evaluate_cosine_similarity(cos_sim, test_labels, reference_labels, k=5)
