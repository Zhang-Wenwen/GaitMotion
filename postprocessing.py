from scipy.ndimage import binary_erosion, binary_dilation, binary_opening, binary_closing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def count_steps(arr):
    up_edges = np.where((arr[:-1] == 0) & (arr[1:] == 1))[0] + 1
    down_edges = np.where((arr[:-1] == 1) & (arr[1:] == 0))[0] + 1
    return up_edges, down_edges

def over_count_steps(array):
    up_edges = (np.roll(array, 1) == 0) & (array == 1)
    down_edges = (np.roll(array, 1) == 1) & (array == 0)
    return min(np.sum(up_edges), np.sum(down_edges))

def miss_over_count(predictions, ground_truth):
    pred_up_edges, pred_down_edges = count_steps(predictions)
    gt_up_edges, gt_down_edges = count_steps(ground_truth)

    # Calculate missing and overcounted steps using numpy arrays
    missing_steps = 0
    overcounted_steps = 0
    tolerance = 256

    # Check for missing steps
    for idx in gt_up_edges:
        if not np.any((idx - tolerance <= pred_up_edges) & (pred_up_edges <= idx + tolerance)):
            missing_steps += 1

    # Check for overcounted steps
    for idx in pred_up_edges:
        if not np.any((idx - tolerance <= gt_up_edges) & (gt_up_edges <= idx + tolerance)):
            overcounted_steps += 1

    return missing_steps, overcounted_steps

def over_steps(predictions, ground_truth):
    prediction_steps_count = over_count_steps(predictions)
    ground_truth_steps_count = over_count_steps(ground_truth)

    # Calculate missing and overcounted steps
    overcounted_steps = max(0, ground_truth_steps_count - prediction_steps_count)
    missing_steps  = max(0, prediction_steps_count - ground_truth_steps_count)

    return missing_steps, overcounted_steps

def compute_metrics(prediction, ground_truth):
    # Ensure binary values
    prediction = np.round(prediction).astype(int)
    ground_truth = np.round(ground_truth).astype(int)

    # Compute confusion matrix elements
    TP = np.sum((prediction == 1) & (ground_truth == 1))
    FP = np.sum((prediction == 1) & (ground_truth == 0))
    FN = np.sum((prediction == 0) & (ground_truth == 1))
    TN = np.sum((prediction == 0) & (ground_truth == 0))

    # Compute Precision, Recall, and F1 Score
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    return precision, recall, f1_score


if __name__ == '__main__':
    read_dir = 'outputs/seg_length=2048/'
    outputs = pd.read_csv(read_dir+'5test.csv', header=None).values
    mask_prd = outputs[:,0]
    labels = outputs[:,1]

    kernel_size = 128
    structuring_element = np.ones(kernel_size)

    # Apply morphological operations
    n=235
    p=0
    opening_result = binary_opening(mask_prd, structure=structuring_element)
    closing_result = binary_closing(opening_result, structure=structuring_element)
    # -------------- overall results caculation: overcounted and missed steps of the subject -------------- 
    # plt.figure(),plt.plot(erosion_result[2048*p:2048*n]),plt.plot(labels[2048*p:2048*n]*1.5, label='true label')
    # plt.savefig(read_dir+'erosion_result.png'), plt.close()
    # plt.figure(),plt.plot(dilation_result[2048*p:2048*n]),plt.plot(labels[2048*p:2048*n]*1.5, label='true label')
    # plt.savefig(read_dir+'dilation_result.png'), plt.close()
    # plt.figure(),plt.plot(opening_result[2048*p:2048*n]),plt.plot(labels[2048*p:2048*n]*1.5, label='true label')
    # plt.savefig(read_dir+'opening_result.png'), plt.close()
    # print("post processing segmentation results")

    # missing_steps, overcounted_steps=miss_over_count(closing_result[2048*p:2048*n], labels[2048*p:2048*n]) #[2048*p:2048*n]
    # print(f"missing_steps: {missing_steps}, overcounted_steps: {overcounted_steps}")

    missing_steps_ov=0
    overcounted_steps_ov = 0
    inde_sep = 1
    for indx in range(0,closing_result.shape[0]//2048,inde_sep):
        missing_steps, overcounted_steps=over_steps(closing_result[2048*indx:2048*(indx+inde_sep)], labels[2048*indx:2048*(indx+inde_sep)])
        missing_steps_ov += missing_steps
        overcounted_steps_ov += overcounted_steps
        # if missing_steps!=0 or overcounted_steps!=0:
        #     print(f"missing_steps: {missing_steps}, overcounted_steps: {overcounted_steps}, index: {indx}")
    print(f"overall missing_steps: {overcounted_steps_ov}, overall overcounted_steps: {missing_steps_ov}")

    # missing_steps, overcounted_steps=over_steps(closing_result[2048*p:2048*n], labels[2048*p:2048*n])
    # print(f"test missing_steps: {missing_steps}, test overcounted_steps: {overcounted_steps}")

    precision, recall, f1_score = compute_metrics(mask_prd, labels)
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

    # np.savetxt(r'./outputs/seg_length=2048/5test_copy.csv',  np.c_[mask_prd[2048*p:2048*n], labels[2048*p:2048*n]], fmt='%i', delimiter=',') 

    plt.figure(),plt.plot(closing_result[2048*p:2048*n],label='prediction'),plt.plot(labels[2048*p:2048*n]*1.5, label='true label')
    plt.savefig(read_dir+'closing_result.png'), plt.show(), plt.close()

    ### ----- get the step counts error for each type of the gait patterns ------
