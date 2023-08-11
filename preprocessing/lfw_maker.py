import pandas as pd
import itertools
import random
from tqdm import tqdm


def generate_pairs(df_path, num_positive_pairs, num_negative_pairs,generate_all=True):
    df = pd.read_csv(df_path)
    pairs = []
    unique_labels = df['label'].unique().tolist()

    print(f"Unique Labels: {unique_labels}")

    # Generate positive pairs (similar images)
    all_positive_pairs = []
    # if generate_all:
    #     num_positive_pairs = 0  # Set to None for generating all pairs
    #     num_negative_pairs = 0

    for label in tqdm(unique_labels):
        label_df = df[df['label'] == label]
        label_images = label_df['image_file'].tolist()

        actual_num_positive_pairs = max(num_positive_pairs, len(list(itertools.combinations(label_images, 2))))
        print(f"Generating {actual_num_positive_pairs} positive pairs for label {label}")

        positive_pairs = list(itertools.combinations(label_images, 2))
        all_positive_pairs.extend(positive_pairs)

    random.shuffle(all_positive_pairs)
    positive_pairs = all_positive_pairs[:num_positive_pairs]

    for img1, img2 in positive_pairs:
        label = df[df['image_file'] == img1]['label'].values[0]
        pairs.append({'image1': img1, 'image2': img2, 'image1_label': label, 'image2_label': label, 'similar': 1})

    # Generate negative pairs (dissimilar images)
    all_images = df['image_file'].tolist()
    num_pairs_created = 0

    while num_pairs_created < num_negative_pairs:
        img1, img2 = random.sample(all_images, 2)
        label1 = df[df['image_file'] == img1]['label'].values[0]
        label2 = df[df['image_file'] == img2]['label'].values[0]

        if label1 != label2:
            pairs.append({'image1': img1, 'image2': img2, 'image1_label': label1, 'image2_label': label2, 'similar': 0})
            num_pairs_created += 1

    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_csv('Dataset/' + df_path.split("/")[1].split("_")[0] + '_pairs.csv', index=False)
    # pairs_df.to_csv('Dataset/test_augmented_pair.csv')

    print("Pairs generated successfully!")
    return pairs_df


def loop_pair_yielding_threshold(df_path, max_dissimilar_pairs=None):
    pairs = []
    df = pd.read_csv(df_path)
    df_len = len(df)
    
    if max_dissimilar_pairs is not None:
        max_dissimilar_pairs = min(max_dissimilar_pairs, df_len * (df_len - 1) // 2)  # Calculate maximum possible
        
    dissimilar_count = 0
    
    for idx1 in range(df_len):
        for idx2 in tqdm(range(idx1 + 1, df_len)): 
            img1_path = df.iloc[idx1, 0]
            img2_path = df.iloc[idx2, 0]
            img1_label = df.iloc[idx1, 1]
            img2_label = df.iloc[idx2, 1]

            if img1_label == img2_label:
                pairs.append({"img1_path": img1_path, "img2_path": img2_path, "similar": 1})
            else:
                if max_dissimilar_pairs is None or dissimilar_count < max_dissimilar_pairs:
                    pairs.append({"img1_path": img1_path, "img2_path": img2_path, "similar": 0})
                    dissimilar_count += 1

    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_csv('LFW.csv', index=False)
