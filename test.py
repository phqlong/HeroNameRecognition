import os
import annoy
from tqdm import tqdm
import torch
import torchvision
from utils import get_embedding, get_hero_path_list, get_test_path_list, path_2_label


ROOT_PATH = os.getcwd()
HERO_NAMES_PATH = f"{ROOT_PATH}/test_data/hero_names.txt"
HERO_IMAGES_DIR = f"{ROOT_PATH}/test_data/hero_images/"
TEST_IMAGES_DIR = f"{ROOT_PATH}/test_data/test_images/"
TEST_LABELS_PATH = f"{ROOT_PATH}/test_data/test.txt"

ANN_PATH = f"{ROOT_PATH}/model/heroes.ann"
MODEL_PATH = f"{ROOT_PATH}/model/model.ckpt"

OUTPUT_PATH = f"{ROOT_PATH}/output/output.txt"

NUM_HEROES = 64

def main():
    hero_images_path_list, hero_images_path_2_label = get_hero_path_list(HERO_IMAGES_DIR)
    test_images_path_list, test_file_2_labels = get_test_path_list(TEST_IMAGES_DIR, TEST_LABELS_PATH)

    # Load ANN 
    annoy_index = annoy.AnnoyIndex(f=512, metric='euclidean')
    annoy_index.load(ANN_PATH)

    # Load model
    # Define the model to be used for feature extraction
    model = torchvision.models.resnet18(pretrained=False)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the last layer (classifier)

    checkpoint = torch.load(MODEL_PATH)
    ckpt_state_dict = {key.replace('model.', ''): value for key, value in checkpoint['state_dict'].items()}

    model.load_state_dict(ckpt_state_dict)
    model.eval()

    acc = 0
    k=1
    predicted_labels = []

    for test_path in tqdm(test_images_path_list):
        query_label = path_2_label(test_path, test_file_2_labels)
        query_embedding = get_embedding(model, test_path, is_test=True)
        nn_indices, nn_scores = annoy_index.get_nns_by_vector(query_embedding, k, include_distances=True)
        nn_labels = [list(hero_images_path_2_label.values())[i%64] for i in nn_indices]

        nn_names = [list(hero_images_path_2_label.values())[i%NUM_HEROES] for i in nn_indices]
        predicted_labels.append(nn_names[0])

        if query_label in nn_labels:
            acc += 1

    print(f"Accuracy: {acc/len(test_images_path_list)}")

    # Write output
    output_dict = {path: label for path, label in zip(test_images_path_list, predicted_labels)}
    with open(OUTPUT_PATH, "w") as f:
        for path, label in output_dict.items():
            f.write(f"{path}\t{label}\n")

if __name__ == "__main__":
    main()