from bounding_box.model import load_weight_model, predict_image,plot_image, get_templated_data, edit_sub_boxes_cut_links, edit_sub_boxes_cut_top
from bounding_box.template import build_templating_data
from bounding_box.ressize import scale_up

###############################################################################################
############# This is an example to show how bounding box predictions look like.###############
###############################################################################################


# Load the model weights from the specified path
# The number of classes is set to 4 because there are 4 main boxes, and the model 'main_bbox_detector_model' was trained with 4 classes.
bbox_model = load_weight_model(r"workspace\models\main_bbox_detector_model.h5",4)

# Make predictions on the input image using the loaded model
# the image cann habe any size
boxes, confidence, classes, ratios = predict_image("../data_zettel/optimal_page/optimal_a4_scale.png", bbox_model)

# Build templating data from XML annotations
org_ms_boxes_person, org_ms_boxes_wohnsitz, org_ms_boxes_ausbildung, org_ms_boxes_wwa, person_class_ids, ausbildung_class_ids, wohnsitz_class_ids, wwa_class_ids, widthOrgImag, heightOrgImag = build_templating_data()

# Get templated data based on the predicted boxes and classes
ausbildung, person, wohnsitz, wwa, best_predicted = get_templated_data(boxes, confidence, classes, org_ms_boxes_person,
                                                                       org_ms_boxes_wohnsitz, org_ms_boxes_ausbildung,
                                                                       org_ms_boxes_wwa, person_class_ids,
                                                                       ausbildung_class_ids, wohnsitz_class_ids,
                                                                       wwa_class_ids)

# There are different ways to adjust the templating.

# First way: cut_top
# ausbildung_cut_top, person_cut_top, wohnsitz_cut_top, wwa_cut_top = edit_sub_boxes_cut_top(ausbildung, person, wohnsitz, wwa)

# Second way: cut_left
ausbildung_cut_links, person_cut_links, wohnsitz_cut_links, wwa_cut_links = edit_sub_boxes_cut_links(ausbildung, person, wohnsitz, wwa)

# Scale up the templating so that it is suitable for the original size of the image.
ausbildung_cut_links, person_cut_links, wohnsitz_cut_links, wwa_cut_links = scale_up(ausbildung_cut_links, person_cut_links, wohnsitz_cut_links, wwa_cut_links, ratios)

# Plot the image, templating, and best predicted box together.
plot_image(("../data_zettel/optimal_page/optimal_a4_scale.png"), ausbildung_cut_links, person_cut_links, wohnsitz_cut_links, wwa_cut_links, best_predicted)