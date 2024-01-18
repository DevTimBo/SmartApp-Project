# from bounding_box.model import load_weight_model, predict_image,plot_image, get_templated_data, edit_sub_boxes_cut_links, edit_sub_boxes_cut_top
# from bounding_box.template import build_templating_data
#
# # inter the path of the model
# bbox_model = load_weight_model(r"C:\Users\alh/PycharmProjects/SmartApp-Project/bounding_box/workspace/models/final_main_model/main_bbox_detector_model.h5",4)
# boxes, confidence, classes = predict_image("workspace/images/test_images/image_0006.jpg", bbox_model)
# org_ms_boxes_person, org_ms_boxes_wohnsitz, org_ms_boxes_ausbildung, org_ms_boxes_wwa, person_class_ids, ausbildung_class_ids, wohnsitz_class_ids, wwa_class_ids, widthOrgImag, heightOrgImag = build_templating_data()
#
# ausbildung, person, wohnsitz, wwa, best_predicted = get_templated_data(boxes, confidence, classes, org_ms_boxes_person,
#                                                                        org_ms_boxes_wohnsitz, org_ms_boxes_ausbildung,
#                                                                        org_ms_boxes_wwa, person_class_ids,
#                                                                        ausbildung_class_ids, wohnsitz_class_ids,
#                                                                        wwa_class_ids)
#
#
# # ausbildung_cut_top, person_cut_top, wohnsitz_cut_top, wwa_cut_top = edit_sub_boxes_cut_top(ausbildung, person, wohnsitz, wwa)
# ausbildung_cut_links, person_cut_links, wohnsitz_cut_links, wwa_cut_links = edit_sub_boxes_cut_links(ausbildung, person, wohnsitz, wwa)
#
# # plot_image(('workspace/images/original/image_0001.jpg'), org_ms_boxes_ausbildung, org_ms_boxes_person, org_ms_boxes_wohnsitz, org_ms_boxes_wwa, best_predicted)
# # plot_image(('workspace/images/original/image_0001.jpg'), ausbildung, person, wohnsitz, wwa, best_predicted)
#
# # plot_image(('workspace/images/original/image_0001.jpg'), ausbildung_cut_top, person_cut_top, wohnsitz_cut_top, wwa_cut_top, best_predicted)
# plot_image(('workspace/images/test_images/image_0006.jpg'), ausbildung_cut_links, person_cut_links, wohnsitz_cut_links, wwa_cut_links, best_predicted)