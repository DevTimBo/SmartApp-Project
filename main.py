import load_data

load_data.read_data()
load_data.split_data()
load_data.create_train_and_test_data()
load_data.get_vocabulary_length_and_clean_labels()
load_data.clean_test_labels()

X_train_img_paths, y_train_labels = load_data.train_img_paths, load_data.train_labels_cleaned
X_test_img_paths, y_test_labels = load_data.test_img_paths, load_data.test_labels_cleaned

